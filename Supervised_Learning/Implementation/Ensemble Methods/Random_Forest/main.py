"""
=============================================================
  main.py  –  Orchestrates the full Random Forest pipeline
=============================================================

  Dataset : Titanic  (GitHub / Kaggle mirror)
  Task    : Binary Classification — Survival prediction
  Model   : Random Forest (Bagging ensemble of Decision Trees)

  Run with:
      python main.py            # full pipeline (with grid search)
      python main.py --fast     # skip grid search for speed

=============================================================
"""

import sys
import os
import argparse

# ── make sure local packages resolve correctly ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from utils.helpers     import print_banner, timer, describe_bagging, ensure_dir
from data.data_loader  import DataLoader
from eda.exploratory_analysis import EDA
from model.random_forest_model import RandomForestModel
from results.visualizer import Visualizer


# ─────────────────────────────────────────────────────────────────────────────
@timer
def step1_load_data() -> DataLoader:
    print_banner("STEP 1 – DATA LOADING & PRE-PROCESSING")
    loader = DataLoader(url=config.DATASET_URL)
    loader.run()
    return loader


@timer
def step2_eda(loader: DataLoader) -> None:
    print_banner("STEP 2 – EXPLORATORY DATA ANALYSIS (EDA)")
    eda = EDA(raw_df=loader.raw_df, clean_df=loader.clean_df)
    eda.run()


@timer
def step3_train(loader: DataLoader, run_gridsearch: bool = True) -> dict:
    print_banner("STEP 3 – MODEL TRAINING")
    describe_bagging()

    rf_obj = RandomForestModel()
    rf_obj.build()

    if run_gridsearch:
        rf_obj.tune(loader.X_train, loader.y_train)
    
    rf_obj.train(loader.X_train, loader.y_train)

    cv_result = rf_obj.cross_validate(
        loader.X_train, loader.y_train,
        cv=config.CV_FOLDS,
    )
    return {"model": rf_obj, "cv": cv_result}


@timer
def step4_results(model_dict: dict, loader: DataLoader) -> None:
    print_banner("STEP 4 – RESULTS & VISUALISATION")
    viz = Visualizer(
        rf_model_obj = model_dict["model"],
        X_test       = loader.X_test,
        y_test       = loader.y_test,
    )
    viz.run(
        X_train    = loader.X_train,
        y_train    = loader.y_train,
        cv_result  = model_dict["cv"],
    )
    print(f"  All outputs saved to: ./{config.OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Random Forest – Titanic pipeline")
    parser.add_argument("--fast", action="store_true",
                        help="Skip grid search for faster execution")
    args = parser.parse_args()

    ensure_dir(config.OUTPUT_DIR)

    print_banner("RANDOM FOREST  (Bagging + Decision Trees)  –  Titanic", width=60)

    loader      = step1_load_data()
    step2_eda(loader)
    model_dict  = step3_train(loader, run_gridsearch=not args.fast)
    step4_results(model_dict, loader)

    print_banner("PIPELINE COMPLETE ✓", width=60)
    print(f"  Outputs directory: {os.path.abspath(config.OUTPUT_DIR)}\n")


if __name__ == "__main__":
    main()
