"""
main.py
-------
Orchestrates the full AdaBoost pipeline:

  1. Data Loading
  2. EDA
  3. Preprocessing
  4. Model Training (AdaBoost variants + GridSearch)
  5. Evaluation & Result Visualisation
"""

import os
import sys

# ── Allow running from any working directory ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_loader   import DataLoader
from eda           import EDA
from preprocessor  import Preprocessor
from model         import AdaBoostTrainer
from evaluator     import Evaluator

OUTPUT_ROOT = "outputs"


def main():
    banner("AdaBoost Heart Disease Classification")

    # ── 1. Data Loading ────────────────────────────────────────────────────────
    banner("Step 1: Data Loading", level=2)
    loader = DataLoader()
    df     = loader.load()
    print(loader.feature_info().to_string(index=False))

    # ── 2. EDA ─────────────────────────────────────────────────────────────────
    banner("Step 2: Exploratory Data Analysis", level=2)
    eda = EDA(df, output_dir=os.path.join(OUTPUT_ROOT, "eda"))
    eda.run()

    # ── 3. Preprocessing ───────────────────────────────────────────────────────
    banner("Step 3: Preprocessing", level=2)
    prep = Preprocessor(test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test = prep.fit_transform(df)

    # ── 4. Training ────────────────────────────────────────────────────────────
    banner("Step 4: Model Training", level=2)
    trainer = AdaBoostTrainer(random_state=42)
    models  = trainer.train(X_train, y_train)

    # ── 5. Evaluation ──────────────────────────────────────────────────────────
    banner("Step 5: Evaluation & Results", level=2)
    evaluator = Evaluator(
        models          = models,
        X_test          = X_test,
        y_test          = y_test,
        feature_names   = prep.feature_names,
        cv_scores       = trainer.cv_scores,
        best_model_name = trainer.best_model_name,
        output_dir      = os.path.join(OUTPUT_ROOT, "results"),
    )
    evaluator.set_train_data(X_train, y_train)
    metrics = evaluator.evaluate()

    # ── Final summary ──────────────────────────────────────────────────────────
    banner("Pipeline Complete!", level=2)
    best_m = metrics[trainer.best_model_name]
    print(f"  Best model : {trainer.best_model_name}")
    print(f"  Accuracy   : {best_m['accuracy']:.4f}")
    print(f"  F1 Score   : {best_m['f1']:.4f}")
    print(f"  ROC-AUC    : {best_m['roc_auc']:.4f}")
    print(f"\n  Outputs saved to ./{OUTPUT_ROOT}/")


# ── Utilities ─────────────────────────────────────────────────────────────────
def banner(text: str, level: int = 1):
    if level == 1:
        line = "=" * 65
        print(f"\n{line}\n  {text}\n{line}")
    else:
        print(f"\n{'─'*65}")
        print(f"  ▶  {text}")
        print("─"*65)


if __name__ == "__main__":
    main()
