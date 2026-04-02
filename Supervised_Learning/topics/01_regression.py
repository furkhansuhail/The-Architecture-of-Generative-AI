"""Module 00 · Regression"""

"""
Regression — Predicting Continuous Values
==========================================

Regression is the foundational task in supervised machine learning.
Before you can understand any specific algorithm, you need to understand
what regression is, why it matters, and the landscape of methods available.
"""

import os
import re
import textwrap

DISPLAY_NAME = "01 · Regression"
ICON         = "📊"
SUBTITLE     = "Predicting continuous numerical values — the foundation of supervised learning"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
What is Regression in Machine Learning?

In machine learning and AI, regression is a type of task where the goal is to predict a continuous numerical value
based on input data.

The core idea is that you're trying to find the relationship between input variables (called features) and a
numerical output. For example:

    * Predict a house price based on square footage, location, and number of bedrooms
    * Predict a person's salary based on years of experience and education
    * Predict tomorrow's temperature based on historical weather data

**Simple Explanation:**

Imagine you're trying to guess how much a house will sell for.
You notice that bigger houses tend to cost more.
If you plotted house size on one axis and price on the other, you'd see a cloud of dots drifting upward — bigger = pricier.

The Key Players:
    1- Independent variable (X) — the thing you know (house size, hours studied, temperature)
    2- Dependent variable (Y) — the thing you're trying to predict (price, test score, ice cream sales)
    3- The regression line — the mathematical relationship between them

Regression draws the best possible line (or curve) through that cloud of dots.
Once you have that line, you can plug in any house size and get a predicted price.

**How it works at a high level:** you feed the model lots of examples with known inputs and outputs, and the model learns
a mathematical function that maps inputs to outputs. Once trained, it can take new inputs it's never seen and produce a
predicted number.

The most classic example is linear regression, which tries to fit a straight line through your data.
If you plotted house size on the x-axis and price on the y-axis, linear regression finds the line that best fits all
your data points, then uses that line to predict prices for new houses.

Other regression algorithms include decision tree regression, random forest regression, and neural network regression —
these can capture more complex, non-linear relationships.

**How is it different from classification?** This is the key distinction in ML.
Regression predicts a number (what will the price be?), while classification predicts a category (is this email spam or not spam?).
If your output is a continuous value, it's a regression problem. If it's a discrete label, it's a classification problem.
So in short — regression is about teaching a model to make numerical predictions by learning patterns from past data.


**Regression Methods — Complete Reference:**

    COMPONENT / METHOD                    BEST FOR                                              OUTPUT TYPE            KEY ASSUMPTION / FEATURE
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    Linear Regression                     Predicting continuous values (straight-line)          Continuous             Linear relationship between X and Y
    Multiple Linear Regression            Multiple input predictors                             Continuous             Linear relationship with multiple predictors
    Polynomial Regression                 Curved / non-linear relationships                     Continuous             Relationship follows polynomial terms (X², X³)
    Ridge Regression (L2)                 Correlated variables; prevents overfitting            Continuous             Shrinks coefficients (L2), keeps all variables
    Lasso Regression (L1)                 Feature selection; sparse models                      Continuous             Can zero-out irrelevant variables
    Elastic Net                           Mix of Ridge and Lasso                                Continuous             Balances L1 and L2 penalties
    Logistic Regression                   Binary classification                                 Probability (0–1)      Categorical outcome (not continuous)
    Multinomial Logistic                  3+ unordered categories                               Probability            Multi-class extension of logistic
    Ordinal Regression                    Ordered categories                                    Ordered category       Respects natural ordering
    Poisson Regression                    Count data (events/time)                              Count (≥0)             Non-negative integer outcome
    Negative Binomial Regression          Count data with high variance                         Count (≥0)             Handles overdispersion
    Quantile Regression                   Predicting percentiles                                Continuous             No strict distribution assumption
    Cox Regression                        Survival / time-to-event                              Hazard rate            Handles censored data
    Tobit Regression                      Censored / bounded outcomes                           Continuous (bounded)   Handles truncated distributions
    Bayesian Regression                   Prior knowledge; uncertainty quantification           Distribution           Returns full probability distributions
    Stepwise Regression                   Automated variable selection                          Continuous             Adds/removes via statistical tests
    Support Vector Regression (SVR)       High-dimensional; robust to outliers                  Continuous             Fits margin "tube"; ignores small errors
    Gaussian Process Regression           Small datasets; uncertainty estimates                 Distribution           Non-parametric; distribution over functions
    Partial Least Squares (PLS)           Many correlated predictors                            Continuous             Projects to uncorrelated components
    Principal Component Regression (PCR)  Multicollinearity                                     Continuous             PCA before regression
    Robust Regression                     Heavy outliers                                        Continuous             Less sensitive than OLS
    Locally Weighted Regression (LOESS)   Flexible local curves                                 Continuous             Local model per neighborhood
    Spline Regression                     Smooth curves with breakpoints                        Continuous             Polynomial pieces joined at knots
    Hierarchical / Multilevel Regression  Grouped / nested data                                 Continuous             Models group structure explicitly
    Random Effects Regression             Panel / longitudinal data                             Continuous             Separates within vs between variation
    Fixed Effects Regression              Panel data; control group differences                 Continuous             Controls time-invariant traits
    Instrumental Variable Regression (IV) Endogeneity / reverse causation                       Continuous             Uses external instrument variable
    Regression Discontinuity (RD)         Causal inference at cutoff                            Continuous             Exploits threshold-based rule
    Geographically Weighted Regression    Spatially varying relationships                       Continuous             Coefficients vary by location
    Neural Network Regression             Complex non-linear patterns                           Continuous             Layered weighted representations
    Decision Tree Regression              Interpretable non-linear models                       Continuous             Rule-based region splits
    Random Forest Regression              High accuracy; complex interactions                   Continuous             Ensemble of decision trees
    Gradient Boosting Regression          High-accuracy tabular predictions                     Continuous             Sequential error-correcting trees
    (XGBoost, LightGBM, etc.)


**Types of Regression Methods:**

A few ways to navigate this list — you can think of them in families:

    * Classic        — Linear, Polynomial, Multiple
    * Regularized    — Ridge, Lasso, Elastic Net
    * Special Output — Logistic, Poisson, Cox, Tobit, Ordinal
    * Robust/Flexible — Quantile, LOESS, Spline, Robust
    * Causal Inference — IV, Regression Discontinuity, Fixed/Random Effects
    * Machine Learning — SVR, Random Forest, Gradient Boosting, Neural Network
"""

OPERATIONS = {
    "1 . Base Linear Regression":{
        "description": " this implements the Ordinary Least Squares (OLS) formula for simple linear regression.",
        "timeout":300,
        "language": "python",
        "code": '''
        
"""
    # =============================================================================
    # Simple Linear Regression from Scratch
    # =============================================================================
    #
    # Linear Regression is a supervised machine learning algorithm that models
    # the relationship between an input variable (x) and a continuous output
    # variable (y) by fitting a straight line through the data points.
    #
    # The line follows the equation:
    #       y = slope * x + intercept
    #
    # To find the best-fit line, this implementation uses the Ordinary Least
    # Squares (OLS) method, which minimizes the sum of squared differences
    # between the actual and predicted y values.
    #
    # The slope and intercept are calculated analytically using:
    #
    #       slope = (n · Σxy − Σx · Σy) / (n · Σx² − (Σx)²)
    #       intercept = (Σy − slope · Σx) / n
    #       
    # where:
    #       n   = number of data points
    #       Σxy = sum of the product of each x and y pair
    #       Σx² = sum of squared x values
    #       Σx  = sum of all x values
    #       Σy  = sum of all y values
    #
    # Once the slope and intercept are known, predictions for new input
    # values can be made by plugging them into:  y = slope * x + intercept
    #
    # Example:
    #       x = [1, 2, 3, 4, 5]  (input/feature)
    #       y = [3, 5, 7, 9, 11] (output/label)
    #       --> The model learns the pattern and predicts y for any new x
    # =============================================================================

"""

import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, x, y):
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x_square = np.sum(x ** 2)

        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_square - sum_x ** 2)
        self.intercept = (sum_y - self.slope * sum_x) / n
        print(f"Slope: {self.slope}, Intercept: {self.intercept}")

    def predict(self, target):
        if self.slope is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        predicted = self.slope * target + self.intercept
        print(f"For target value {target}, prediction is {predicted:.2f}.")
        return predicted

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

model = LinearRegression()
model.fit(x, y)

# Target value is hardcoded here — input() cannot be used in a subprocess runner
target = 6
model.predict(target)
        
        '''


    }

}


COMPLEXITY = """\
+--------------------------------------+-------------------------------------------------------+----------------------+----------------------------------------------+
| COMPONENT / METHOD                   | BEST FOR                                              | OUTPUT TYPE          | KEY ASSUMPTION / FEATURE                     |
+--------------------------------------+-------------------------------------------------------+----------------------+----------------------------------------------+
| Linear Regression                    | Predicting continuous values (straight-line)          | Continuous           | Linear relationship between X and Y          |
| Multiple Linear Regression           | Multiple input predictors                             | Continuous           | Linear relationship with multiple predictors |
| Polynomial Regression                | Curved / non-linear relationships                     | Continuous           | Relationship follows polynomial terms (X², X³)|
| Ridge Regression (L2)                | Correlated variables; prevents overfitting            | Continuous           | Shrinks coefficients (L2), keeps all variables|
| Lasso Regression (L1)                | Feature selection; sparse models                      | Continuous           | Can zero-out irrelevant variables            |
| Elastic Net                          | Mix of Ridge and Lasso                                | Continuous           | Balances L1 and L2 penalties                 |
| Logistic Regression                  | Binary classification                                 | Probability (0–1)    | Categorical outcome (not continuous)         |
| Multinomial Logistic                 | 3+ unordered categories                               | Probability          | Multi-class extension of logistic            |
| Ordinal Regression                   | Ordered categories                                    | Ordered category     | Respects natural ordering                    |
| Poisson Regression                   | Count data (events/time)                              | Count (≥0)           | Non-negative integer outcome                 |
| Negative Binomial Regression         | Count data with high variance                         | Count (≥0)           | Handles overdispersion                       |
| Quantile Regression                  | Predicting percentiles                                | Continuous           | No strict distribution assumption            |
| Cox Regression                       | Survival / time-to-event                              | Hazard rate          | Handles censored data                        |
| Tobit Regression                     | Censored / bounded outcomes                           | Continuous (bounded) | Handles truncated distributions              |
| Bayesian Regression                  | Prior knowledge; uncertainty quantification           | Distribution         | Returns full probability distributions       |
| Stepwise Regression                  | Automated variable selection                          | Continuous           | Adds/removes via statistical tests           |
| Support Vector Regression (SVR)      | High-dimensional; robust to outliers                  | Continuous           | Fits margin "tube"; ignores small errors     |
| Gaussian Process Regression          | Small datasets; uncertainty estimates                 | Distribution         | Non-parametric; distribution over functions  |
| Partial Least Squares (PLS)          | Many correlated predictors                            | Continuous           | Projects to uncorrelated components          |
| Principal Component Regression (PCR) | Multicollinearity                                     | Continuous           | PCA before regression                        |
| Robust Regression                    | Heavy outliers                                        | Continuous           | Less sensitive than OLS                      |
| Locally Weighted Regression (LOESS)  | Flexible local curves                                 | Continuous           | Local model per neighborhood                 |
| Spline Regression                    | Smooth curves with breakpoints                        | Continuous           | Polynomial pieces joined at knots            |
| Hierarchical / Multilevel Regression | Grouped / nested data                                 | Continuous           | Models group structure explicitly            |
| Random Effects Regression            | Panel / longitudinal data                             | Continuous           | Separates within vs between variation        |
| Fixed Effects Regression             | Panel data; control group differences                 | Continuous           | Controls time-invariant traits               |
| Instrumental Variable Regression (IV)| Endogeneity / reverse causation                       | Continuous           | Uses external instrument variable            |
| Regression Discontinuity (RD)        | Causal inference at cutoff                            | Continuous           | Exploits threshold-based rule                |
| Geographically Weighted Regression   | Spatially varying relationships                       | Continuous           | Coefficients vary by location                |
| Neural Network Regression            | Complex non-linear patterns                           | Continuous           | Layered weighted representations             |
| Decision Tree Regression             | Interpretable non-linear models                       | Continuous           | Rule-based region splits                     |
| Random Forest Regression             | High accuracy; complex interactions                   | Continuous           | Ensemble of decision trees                   |
| Gradient Boosting Regression         | High-accuracy tabular predictions                     | Continuous           | Sequential error-correcting trees            |
| (XGBoost, LightGBM, etc.)            |                                                       |                      |                                              |
+--------------------------------------+-------------------------------------------------------+----------------------+----------------------------------------------+
"""

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
        main_script = None

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
        from Supervised_Learning.visuals.regression_visual import (   # ← match your exact folder casing
            REG_VISUAL_HTML,
            REG_VISUAL_HEIGHT,
        )
        visual_html   = REG_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = REG_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[01_linear_regression.py] Could not load visual: {e}", stacklevel=2)

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

