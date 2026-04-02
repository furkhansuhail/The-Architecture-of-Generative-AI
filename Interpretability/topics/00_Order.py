
import textwrap
import re

TOPIC_NAME   = "Order for Interpretability"
DISPLAY_NAME = "00 · Order for Interpretability"
ICON         = "⚖️"
SUBTITLE     = "Order for Learning Training Core with priorities"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### Interpretability in AI/ML

Interpretability means understanding what's happening inside a model — why it made a specific decision, 
what patterns it learned, and how it arrives at its outputs.

## The Core Problem

A neural network with millions or billions of parameters is essentially a black box. 
You give it an input, it gives you an output, but the path between the two is a sea of matrix multiplications and 
non-linear activations that no human can read directly. Interpretability is the field of techniques trying to open that box.

## Two Levels of Understanding

Local interpretability asks: why did the model make this specific prediction? For example — why did it classify this 
X-ray as cancerous, or why did it approve this loan application?
Global interpretability asks: what has the model learned in general? What features matter most across all predictions? 
What concepts has it internalized?

## Why It Matters
The stakes differ depending on context. In a movie recommendation system, a wrong prediction costs nothing. 
But in medical diagnosis, credit scoring, criminal sentencing, or autonomous vehicles, you need to be able to audit, 
challenge, and trust the model's reasoning. Interpretability is also essential for catching bias — a model might 
perform well on benchmarks while having learned to discriminate by race or gender through proxy variables, 
and you can't catch that without looking inside.

For AI safety specifically, interpretability is critical for understanding whether a powerful model has learned the 
goals and values you intended, or something subtly different.

## Common Techniques

SHAP and LIME are probably the most widely used in practice. They work by perturbing inputs and measuring how the output
changes, attributing importance scores to each input feature. If a model predicts loan default, SHAP might tell you that 
income contributed -0.3 (reducing risk) and missed payments contributed +0.8 (increasing risk) for a particular applicant.

Attention visualization applies to transformers — you can inspect which tokens the model attended to when producing an 
output, though researchers debate whether attention weights actually reflect the model's reasoning or are just a side-effect.

Probing classifiers train a small classifier on top of a model's internal representations to test whether a specific 
concept (e.g. "is this token a verb?") is encoded in the activations.

Mechanistic interpretability is the deepest and most ambitious approach — instead of probing from the outside, it tries
to reverse-engineer the actual algorithms learned by the network. Researchers like Chris Olah have identified specific 
circuits inside neural networks that perform recognizable operations like curve detection or induction heads in transformers. 
Anthropic has done significant work in this space.

Saliency maps (common in computer vision) highlight which pixels of an image most influenced the prediction — 
useful for checking whether a tumor detector is actually looking at the tumor or at an irrelevant artifact like a 
scanner watermark.

## Interpretability vs. Explainability
These terms are often used interchangeably but some researchers distinguish them. 
Interpretability is an intrinsic property — how inherently understandable the model's structure is (a decision tree is highly interpretable; 
a 70B parameter transformer is not). Explainability refers to post-hoc methods applied to a black box to generate 
human-readable explanations after the fact. The explanations may or may not faithfully reflect the true internal computation.

## The Hard Unsolved Problem
The deepest challenge is that most current techniques give you correlations and approximations, 
not true causal accounts of what the model is doing. An explanation like "the model focused on these words" might be 
a faithful description of attention weights but not of the actual computation that produced the output. 
Mechanistic interpretability aims to close this gap but remains very much a research frontier, especially as models grow larger.

In short — interpretability is about earning the right to trust a model, not just on its aggregate accuracy, 
but on the specific reasoning it uses case by case.


       
    interpretability/
    ├── scope_and_taxonomy/
    ├── intrinsic_models/
    │   ├── linear_models
    │   ├── decision_trees
    │   └── rule_lists
    ├── post_hoc_agnostic/
    │   ├── feature_importance/
    │   │   ├── permutation_importance
    │   │   └── integrated_gradients
    │   ├── local_explanations/
    │   │   ├── SHAP
    │   │   ├── LIME
    │   │   └── anchors
    │   └── global_visualizations/
    │       ├── PDP
    │       ├── ICE
    │       └── ALE
    ├── nn_specific_xai/
    │   ├── gradient_based/
    │   │   ├── saliency_maps
    │   │   ├── GradCAM
    │   │   ├── LRP
    │   │   └── DeepLIFT
    │   └── attention_analysis/
    │       └── BERTViz
    ├── counterfactual/
    ├── concept_based/
    ├── mechanistic_interpretability/
    │   ├── circuits
    │   ├── probing_classifiers
    │   └── superposition
    ├── model_calibration
    ├── evaluation_of_explanations/
    └── fairness_and_bias/

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }



