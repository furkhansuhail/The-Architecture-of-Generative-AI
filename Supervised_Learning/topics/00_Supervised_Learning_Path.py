"""Module 00 · Supervised Learning — Algorithm Reference Table"""

"""
Supervised Learning — Complete Algorithm Reference (ASIC Table)
===============================================================

A single-source reference for every major supervised learning algorithm.
Each entry follows the ASIC format:

    A — Algorithm name and family
    S — Summary (what problem it solves and how it works)
    I — Input / Output types
    C — Characteristics (assumptions, strengths, and limitations)

Use this as a map before diving into any individual algorithm module.
"""

import os
import re
import textwrap

DISPLAY_NAME = "00 · Supervised Learning Path"
ICON         = "🗺️"
SUBTITLE     = "Algorithm · Summary · Input/Output · Characteristics — the complete supervised learning map"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
What is Supervised Learning?

Supervised learning is the branch of machine learning where the model learns from labelled examples.
Each training example consists of an input X and a known output y. The model learns a function
f(X) → y that generalises to new, unseen inputs.

Two fundamental tasks:
    * Regression    — predict a continuous number (house price, temperature, salary)
    * Classification — predict a discrete category (spam/not spam, dog/cat, diagnosis)

The ASIC Table below maps every major supervised algorithm across four dimensions:
    A — Algorithm         what the model is called and which family it belongs to
    S — Summary           what problem it solves and the core idea behind it
    I — Input / Output    what data it expects and what it produces
    C — Characteristics   key assumptions, strengths, and limitations


##### FAMILY 1 — LINEAR MODELS
    
    These models assume the relationship between inputs and output can be captured by
    a weighted sum of features. Fast, interpretable, and highly scalable.
    

    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
    | ALGORITHM                  | SUMMARY                                                     | INPUT / OUTPUT                       | CHARACTERISTICS                                                                       |
    +----------------------------+--------------------------------------------------------------+-------------------------------------+---------------------------------------------------------------------------------------+
    | Linear Regression          | Fits a straight line (or hyperplane) through data           | Continuous features           →      | Assumes linear relationship; closed-form solution (Normal Equation)                   |
    |                            | by minimising Mean Squared Error.                           | Continuous output                    | or Gradient Descent; fast and interpretable; fails on non-linear data                 |
    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
    | Multiple Linear Regression | Extension of linear regression to many features.            | Continuous features (p > 1)   →      | Same assumptions as linear regression; suffers from multicollinearity;                |
    |                            | Finds the best-fit hyperplane in p-dimensional space.       | Continuous output                    | use Ridge/Lasso when features are correlated                                          |
    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
    | Polynomial Regression      | Adds powers of features (X², X³, ...) to capture            | Continuous features           →      | Still linear in the weights (linear model); can overfit badly with                    |
    |                            | curved relationships. Bends the regression line.            | Continuous output                    | high-degree polynomials; choose degree carefully via cross-validation                 |
    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
    | Ridge Regression (L2)      | Linear regression with an L2 penalty (λ Σ wᵢ²)              | Continuous features           →      | Shrinks all weights toward zero; never zeros them out; handles                        |
    |                            | added to the loss. Controls overfitting.                    | Continuous output                    | multicollinearity; hyperparameter λ tuned by cross-validation                         |
    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
    | Lasso Regression (L1)      | Linear regression with an L1 penalty (λ Σ |wᵢ|)             | Continuous features           →      | Produces sparse solutions — zeros out irrelevant features;                            |
    |                            | added to the loss. Performs feature selection.              | Continuous output                    | automatic feature selection; picks one of correlated features arbitrarily             |
    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
    | Elastic Net                | Combines L1 and L2 penalties:                               | Continuous features           →      | Inherits sparsity from Lasso and grouping from Ridge; two                             |
    |                            | λ [ρ Σ|wᵢ| + (1-ρ)/2 Σwᵢ²]. Best of both.                   | Continuous output                    | hyperparameters (λ, ρ); preferred when features are correlated + many                 |
    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
    | Logistic Regression        | Applies sigmoid to a linear combination to predict          | Binary/multi-class features   →      | Despite the name, it is a classifier; decision boundary is linear;                    |
    |                            | class probabilities. Threshold at 0.5 for class.            | Probability (0–1) / Class            | outputs calibrated probabilities; extend to multi-class via softmax                   |
    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
    | Bayesian Linear Regression | Treats weights as probability distributions rather          | Continuous features           →      | Returns full posterior distribution over predictions, not a point                     |
    |                            | than fixed values. Incorporates prior beliefs.              | Distribution over outputs            | estimate; handles small/noisy data well; computationally expensive                    |
    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
    | Quantile Regression        | Predicts a specific quantile (e.g. median, 90th             | Continuous features           →      | No distributional assumptions; robust to outliers; useful when                        |
    |                            | percentile) rather than the conditional mean.               | Continuous output (percentile)       | you care about extremes (e.g. worst-case delivery time)                               |
    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
    | Robust Regression          | Minimises a loss that is less sensitive to outliers         | Continuous features           →      | Iteratively reweighted least squares (IRLS); slower than OLS;                         |
    |                            | (e.g. Huber loss, Tukey bisquare).                          | Continuous output                    | use when genuine outliers exist, not just high-leverage points                        |
    +----------------------------+-------------------------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------+
 
 
##### FAMILY 2 — PROBABILISTIC / GENERATIVE MODELS
 
These models learn the underlying data distribution P(X | y) and use Bayes' theorem
to derive P(y | X). They classify by asking: "which class was most likely to generate
this input?"

    ALGORITHM                       SUMMARY                                                  INPUT / OUTPUT                  CHARACTERISTICS
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Gaussian Naive Bayes            Assumes features are conditionally independent given      Continuous features →           "Naive" = independence assumption is rarely true but often works;
                                    the class, and follow a Gaussian distribution.            Class label                     very fast; works well on small data; good baseline for NLP tasks

    Multinomial Naive Bayes         Same independence assumption but for count data           Count / frequency features →    Classic text classification model (bag of words);
                                    (word frequencies, document counts).                      Class label                     features must be non-negative; fast and interpretable

    Bernoulli Naive Bayes           Models binary features (present / absent).               Binary features →               Used for binary document classification;
                                    Penalises absent features explicitly.                     Class label                     more accurate than Multinomial when features are rare

    Linear Discriminant Analysis    Finds a linear projection that maximises the ratio of     Continuous features →           Assumes Gaussian class distributions with equal covariance;
    (LDA)                           between-class to within-class variance.                   Class label                     reduces dimensionality as a side effect; good for small datasets

    Quadratic Discriminant Analysis Same as LDA but allows each class to have its own        Continuous features →           More flexible than LDA (different covariances per class);
    (QDA)                           covariance matrix. Non-linear decision boundary.          Class label                     requires more data; can overfit with many features



##### FAMILY 3 — INSTANCE-BASED / LAZY LEARNERS

These models do not learn an explicit function during training. Instead they memorise
the training data and make predictions at inference time by looking up similar examples.

    ALGORITHM                       SUMMARY                                                  INPUT / OUTPUT                  CHARACTERISTICS
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    K-Nearest Neighbors (KNN)       For a new point, find the k closest training examples    Any features (needs distance) → No training cost; O(n) prediction cost; very sensitive to
    — Classification                and vote on the class by majority.                       Class label                     feature scaling; breaks down in high dimensions (curse of dimensionality)

    K-Nearest Neighbors (KNN)       For a new point, find the k closest training examples    Any features (needs distance) → Same as classification variant; k controls bias-variance tradeoff;
    — Regression                    and average their target values.                         Continuous output               small k = high variance, large k = high bias


##### FAMILY 4 — SUPPORT VECTOR MACHINES

SVMs find the decision boundary that maximises the margin between classes (or, for
regression, fits the most data within a "tube" around the line). The kernel trick
allows them to operate in high-dimensional feature spaces without computing those
spaces explicitly.

    ALGORITHM                       SUMMARY                                                  INPUT / OUTPUT                  CHARACTERISTICS
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    SVM — Linear Kernel             Finds the maximum-margin linear hyperplane separating     Continuous/binary features →    Works best when data is linearly separable; fast; scales to
                                    two classes. Defined by support vectors only.             Class label                     large datasets with SGD (LinearSVC); memory efficient

    SVM — RBF Kernel                Uses Radial Basis Function kernel to project data into    Any features →                  Most popular SVM variant; handles non-linear boundaries;
                                    an implicit infinite-dimensional space.                   Class label                     hyperparameters C and γ require tuning; slow on large datasets

    SVM — Polynomial Kernel         Uses polynomial kernel k(x,z) = (xᵀz + c)^d to          Any features →                  Explicitly models feature interactions up to degree d;
                                    capture interactions between features.                    Class label                     degree d is a hyperparameter; can overfit with high d

    Support Vector Regression       Fits data within an ε-insensitive tube. Ignores          Continuous features →           Robust to outliers outside the tube; hyperparameters ε, C;
    (SVR)                           small errors; penalises only large deviations.           Continuous output               works well in high-dimensional spaces; slow for large n


##### FAMILY 5 — TREE-BASED MODELS

Decision trees recursively partition the feature space with axis-aligned splits.
They are highly interpretable but tend to overfit. Ensembles of trees (Random Forests,
Gradient Boosting) are among the strongest models for tabular data.

    ALGORITHM                       SUMMARY                                                  INPUT / OUTPUT                  CHARACTERISTICS
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Decision Tree Classifier        Splits data recursively on features that best             Any features (handles mixed) → Highly interpretable; can visualise the tree; overfits without
                                    separate classes (Gini impurity or entropy).              Class label                     pruning; handles non-linear boundaries; no feature scaling needed

    Decision Tree Regressor         Same splitting procedure but at each leaf predicts        Any features (handles mixed) → Same interpretability benefits; predictions are step functions
                                    the mean of the training examples in that region.         Continuous output               (piecewise constant); prone to overfitting on training data

    Random Forest Classifier        Trains many decision trees on bootstrapped subsets        Any features →                  Strong out-of-box performance; resistant to overfitting;
                                    of data + random feature subsets. Votes on class.         Class label                     parallelisable; provides feature importances; ~hundreds of trees

    Random Forest Regressor         Same as Random Forest Classifier but averages            Any features →                  Same properties; predictions are averages of many trees;
                                    the continuous predictions from each tree.                Continuous output               smoother than a single tree; may underperform on sparse data

    Extra Trees (Extremely          Like Random Forest but splits are chosen at random        Any features →                  Lower variance than Random Forest; faster training;
    Randomized Trees)               rather than finding the best split point.                Class / Continuous output       slightly higher bias; good regularisation via randomness

    Gradient Boosting Classifier    Builds trees sequentially. Each tree corrects the         Any features →                  Strong benchmark for tabular data; requires careful tuning;
                                    residual errors of the previous ensemble.                 Class label                     learning rate, n_estimators, and depth are key hyperparameters

    Gradient Boosting Regressor     Same sequential boosting process targeting the            Any features →                  Same strengths; can overfit without early stopping;
                                    regression residuals.                                     Continuous output               widely used in Kaggle competitions for tabular regression

    XGBoost                         Optimised implementation of gradient boosting with        Any features →                  Regularisation built in (L1/L2); handles missing values natively;
                                    regularisation, parallelism, and cache optimisation.      Class / Continuous output       column subsampling; industry standard for tabular data

    LightGBM                        Gradient boosting using histogram-based splitting         Any features (categorical OK) → Leaf-wise tree growth (faster than level-wise); very fast on large
                                    and leaf-wise tree growth. Microsoft-developed.           Class / Continuous output       datasets; may overfit more easily; handles large feature counts well

    CatBoost                        Gradient boosting with native handling of categorical     Mixed features (cats welcome) → Ordered boosting reduces target leakage; strong on datasets with
                                    features using target statistics. Yandex-developed.       Class / Continuous output       many categorical columns; slower training than LightGBM


##### FAMILY 6 — ENSEMBLE META-METHODS

These are strategies for combining multiple models rather than specific algorithms.
They generally outperform any single model by reducing variance (Bagging), bias
(Boosting), or both (Stacking).

    ALGORITHM                       SUMMARY                                                  INPUT / OUTPUT                  CHARACTERISTICS
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Bagging                         Trains n models on bootstrapped (sampled with             Any base learner →              Reduces variance; best with high-variance base learners (e.g. deep
                                    replacement) subsets of data. Averages/votes results.     Class / Continuous output       trees); Random Forest is the most famous special case

    Boosting                        Trains models sequentially. Each new model focuses        Any base learner (weak) →       Reduces bias; best with weak learners (e.g. shallow trees);
                                    on the examples the previous models got wrong.            Class / Continuous output       AdaBoost, Gradient Boosting, XGBoost are all instances of boosting

    AdaBoost                        Assigns higher weights to misclassified examples          Weak classifier →               One of the first successful boosting algorithms; sensitive to noisy
                                    at each round so subsequent learners focus on them.       Class label                     data and outliers; slow compared to gradient boosting variants

    Stacking (Stacked Generalisation) Trains a meta-model (Level 2) on the out-of-fold    Any base learners (Level 1) →   Powerful but complex; risk of overfitting if not done with
                                    predictions of base models (Level 1).                    Class / Continuous output       cross-validation; Level 2 model is often simple (logistic regression)

    Voting Classifier               Combines predictions from multiple different models       Any classifiers →               Hard voting (majority class) or soft voting (average probabilities);
                                    by majority vote or averaging probabilities.              Class label                     easy to implement; models should be diverse to be effective

    Voting Regressor                Averages predictions from multiple regression models.     Any regressors →                Reduces variance; works best when base models make uncorrelated
                                    Each model contributes equally to the average.            Continuous output               errors; unlike stacking, no meta-model is needed


##### FAMILY 7 — NEURAL NETWORKS

Neural networks stack layers of linear transformations followed by non-linear
activation functions. They can approximate any continuous function (universal
approximation theorem) and are the foundation of modern deep learning.

    ALGORITHM                       SUMMARY                                                  INPUT / OUTPUT                  CHARACTERISTICS
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Perceptron                      The simplest neural unit: weighted sum of inputs          Binary features →               First supervised learning algorithm (Rosenblatt, 1958);
                                    followed by a step activation function.                   Binary class (+1 / -1)          only converges if data is linearly separable; ancestor of all NNs

    Multi-Layer Perceptron          Fully-connected layers with non-linear activations        Any features (tabular) →        Universal approximator; trained by backpropagation;
    (MLP / Feedforward NN)          (ReLU, sigmoid). Can model any continuous function.       Class / Continuous output       requires feature scaling; many hyperparameters; prone to overfitting

    Convolutional Neural Network    Applies learned filters (kernels) to local regions        Grid data (images, audio) →     Weight sharing drastically reduces parameters; translation invariant;
    (CNN)                           of the input. Detects local patterns hierarchically.      Class / Dense output            state-of-the-art for image classification; needs large datasets

    Recurrent Neural Network        Processes sequences step by step, maintaining a          Sequential data (text, time) →  Suffers from vanishing/exploding gradients; largely replaced by
    (RNN)                           hidden state that captures temporal dependencies.         Class / Sequence output         LSTMs and Transformers in modern practice

    Long Short-Term Memory          Adds gating mechanisms (input, forget, output gates)      Sequential / Time-series →      Solves vanishing gradient problem; can learn long-range dependencies;
    (LSTM)                          to RNN to selectively remember or forget information.     Class / Sequence output         computationally expensive; slower to train than Transformers

    Gated Recurrent Unit (GRU)      Simplified LSTM with fewer gates (update + reset).        Sequential / Time-series →      Fewer parameters than LSTM; often comparable performance;
                                    More efficient with similar capabilities.                 Class / Sequence output         good default for sequence modelling when speed matters

    Transformer                     Uses self-attention to weigh the importance of every      Sequential / Any features →     No recurrence = highly parallelisable; scales with data and compute;
                                    position in the input relative to every other.            Class / Sequence / Continuous   foundation of BERT, GPT, ViT; requires large training data

    Residual Network (ResNet)       Adds skip connections that bypass one or more layers.     Image features (CNN-based) →    Allows very deep networks (100+ layers) without vanishing gradients;
                                    Learns residual functions instead of direct mappings.     Class label                     skip connections are now standard practice in deep learning

    Graph Neural Network (GNN)      Operates on graph-structured data. Aggregates             Graph (nodes, edges, features) → Models relational data; used in molecular biology, social networks,
                                    information from a node's neighbours iteratively.         Node / Graph / Edge output      recommendation systems; challenging to scale to very large graphs



##### FAMILY 8 — SPECIALISED REGRESSION METHODS

These handle output types or data structures that standard linear regression cannot
address: count data, bounded outputs, survival times, grouped data, and more.

    ALGORITHM                       SUMMARY                                                  INPUT / OUTPUT                  CHARACTERISTICS
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Poisson Regression              Models count data using a log link function.              Continuous features →           Output is always non-negative integer; assumes mean = variance
                                    Part of the Generalised Linear Model (GLM) family.        Count output (≥0)               (Poisson assumption); use Negative Binomial if data is overdispersed

    Negative Binomial Regression    Extension of Poisson regression for overdispersed         Continuous features →           Adds a dispersion parameter to handle variance > mean;
                                    count data (variance > mean).                             Count output (≥0)               better than Poisson when data shows clustering or heavy tails

    Ordinal Regression              Predicts ordered categories                               Continuous features →           Respects natural ordering (e.g. mild < moderate < severe);
                                    (e.g. star ratings, severity levels).                     Ordered category                doesn't assume equal spacing between categories

    Multinomial Logistic Regression Extension of logistic regression to 3+ unordered         Continuous features →           Uses softmax; one set of weights per class; assumes linear
                                    classes. Predicts a probability vector over all classes.  Probability vector / Class      decision boundaries; scalable; good multi-class baseline

    Tobit Regression                Models outcomes that are censored — observed only         Continuous features →           Handles floor or ceiling effects in the data; often used in
                                    above or below a threshold (e.g. wages, limits).          Continuous (bounded)            economics; maximum likelihood estimation

    Cox Proportional Hazards        Models time until an event (death, failure, churn).       Time + event features →         Handles censored observations; assumes proportional hazards;
    (Survival Regression)           Estimates the hazard function over time.                  Hazard rate / Survival curve    widely used in clinical trials and reliability engineering

    Gaussian Process Regression     Places a prior distribution over functions. The           Any features (small n) →        Non-parametric; provides uncertainty estimates; exact inference
                                    posterior is also a Gaussian process.                     Distribution over outputs       is O(n³) — only practical for small datasets (n < ~10,000)

    Spline Regression               Fits piecewise polynomial curves joined smoothly          Continuous features →           More flexible than polynomial; knot placement is key;
                                    at "knot" points. Can use cubic splines or B-splines.     Continuous output               interpretable; generalised additive models (GAMs) build on this

    Locally Weighted Regression     Fits a local linear model around each prediction          Continuous features →           No global model; adapts to local structure; computationally
    (LOESS / LOWESS)                point using a distance-weighted kernel.                   Continuous output               expensive at prediction time; good for data exploration

    Hierarchical / Multilevel       Models data with nested structure (students within        Grouped/nested features →       Accounts for group-level variance; partial pooling between groups;
    Regression                      schools, patients within hospitals).                      Continuous output               requires enough groups to estimate random effects reliably

    Fixed Effects Regression        Controls for all time-invariant individual                Panel data (repeated measures)→  Eliminates confounding from stable individual traits; cannot
                                    characteristics using individual dummies or demeaning.    Continuous output               estimate effects of time-invariant variables (e.g. gender)

    Instrumental Variable (IV)      Uses an external instrument Z (correlated with X         Continuous features + instrument Addresses endogeneity / reverse causality; valid instrument is
    Regression                      but not with the error) to identify causal effects.       → Continuous output             hard to find; weak instruments produce unreliable estimates

    Regression Discontinuity (RD)   Exploits a sharp threshold rule to estimate causal        Continuous running variable →   Local randomisation around cutoff; treatment effect valid only
                                    effects just above and below the cutoff.                  Continuous output               near the threshold; requires large sample near the cut-off


##### FAMILY 9 — DIMENSIONALITY-REDUCTION ASSISTED REGRESSION

These methods reduce dimensionality before or during regression, making them
powerful when the number of features (p) is much larger than the number of
observations (n), or when features are highly correlated.

    ALGORITHM                       SUMMARY                                                  INPUT / OUTPUT                  CHARACTERISTICS
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Principal Component Regression  Runs PCA first to remove multicollinearity, then         High-dimensional features →     Interpretability is lost (coefficients are on components, not
    (PCR)                           runs OLS on the principal components.                    Continuous output               features); choosing the number of components is the key decision

    Partial Least Squares (PLS)     Like PCR but finds components that maximise              High-dimensional features →     Especially powerful in chemometrics / spectroscopy;
                                    covariance with y (not just variance of X).              Continuous output               components directly related to the target; better than PCR when
                                                                                                                             p >> n and X and y are correlated


##### HOW TO CHOOSE — A DECISION GUIDE

    ┌────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                            │
    │  What is your output?                                                                      │
    │                                                                                            │
    │  Continuous number       → Start with Linear Regression, then try Random Forest / XGBoost  │
    │  Binary class            → Start with Logistic Regression, then try SVM / Gradient Boost   │
    │  Multi-class             → Multinomial Logistic, Random Forest, or Neural Network          │
    │  Count data              → Poisson or Negative Binomial Regression                         │
    │  Ordered categories      → Ordinal Regression                                              │
    │  Time-to-event           → Cox Proportional Hazards                                        │
    │  Sequences / Text        → LSTM, GRU, or Transformer                                       │
    │  Images                  → CNN or ViT (Vision Transformer)                                 │
    │  Graphs / Networks       → Graph Neural Network                                            │
    │                                                                                            │
    │  How much data do you have?                                                                │
    │                                                                                            │
    │  Small (n < 1,000)       → Linear models, Naive Bayes, SVM, Gaussian Process               │
    │  Medium (1k – 100k)      → Random Forest, Gradient Boosting, SVM                           │
    │  Large (100k+)           → Gradient Boosting (XGBoost/LightGBM), Neural Networks           │
    │                                                                                            │
    │  Do you need interpretability?                                                             │
    │                                                                                            │
    │  High interpretability   → Linear/Logistic Regression, Decision Tree, LDA                  │
    │  Moderate                → Random Forest (feature importances), Lasso (sparse weights)     │
    │  Prediction only         → Gradient Boosting, Neural Networks, Stacking                    │
    │                                                                                            │
    │  How many features do you have?                                                            │
    │                                                                                            │
    │  Many correlated (p >> n) → Ridge, Lasso, ElasticNet, PLS, PCR                             │
    │  Many with irrelevant    → Lasso (feature selection), Tree-based models                    │
    │  Mixed types (cat + num) → Gradient Boosting (native), CatBoost                            │
    │                                                                                            │
    └────────────────────────────────────────────────────────────────────────────────────────────┘


##### QUICK REFERENCE — ALGORITHM FAMILIES AT A GLANCE

    Family                           Algorithms
    ─────────────────────────────────────────────────────────────────────────────────────────────
    Linear Models                    Linear Regression, Logistic Regression, Ridge, Lasso,
                                     Elastic Net, Polynomial, Quantile, Robust, Bayesian LR

    Probabilistic / Generative       Gaussian Naive Bayes, Multinomial NB, Bernoulli NB,
                                     LDA, QDA

    Instance-Based                   KNN Classifier, KNN Regressor

    Support Vector Machines          SVM (Linear, RBF, Polynomial), SVR

    Tree-Based                       Decision Tree, Random Forest, Extra Trees,
                                     Gradient Boosting, XGBoost, LightGBM, CatBoost

    Ensemble Meta-Methods            Bagging, Boosting, AdaBoost, Stacking, Voting

    Neural Networks                  Perceptron, MLP, CNN, RNN, LSTM, GRU,
                                     Transformer, ResNet, GNN

    Specialised Regression           Poisson, Negative Binomial, Ordinal, Multinomial Logistic,
                                     Tobit, Cox (Survival), Gaussian Process, Spline, LOESS,
                                     Hierarchical, Fixed Effects, IV Regression, RD Regression

    Dimensionality-Reduction Aided   PCR (Principal Component Regression), PLS
    ─────────────────────────────────────────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    {},
    }