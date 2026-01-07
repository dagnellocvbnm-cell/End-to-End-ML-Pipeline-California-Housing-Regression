from __future__ import annotations

import os
import joblib

from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CFG
from src.data import load_and_split

def build_model(*, ridge_alpha: float) -> Pipeline:
    """
    Builds a leakage-safe sklearn Pipeline.

    Pipeline stages:
      1) Impute missing values (median)
      2) Standardize features (mean 0, std 1)
      3) Fit Ridge regression model
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=ridge_alpha)),
    ])

def main() -> None:
    split = load_and_split(test_size=CFG.test_size, random_seed=CFG.random_seed)

    # Baseline 0: "dumb" predictor (mean of y). Sets a must-beat bar.
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(split.X_train, split.y_train)

    # Baseline 1: Ridge regression in a preprocessing pipeline.
    pipeline = build_model(ridge_alpha=CFG.ridge_alpha)
    pipeline.fit(split.X_train, split.y_train)

    os.makedirs(os.path.dirname(CFG.model_path), exist_ok=True)
    joblib.dump(
        {"dummy": dummy, "pipeline": pipeline, "feature_names": split.feature_names},
        CFG.model_path,
    )

    print(f"Saved models to: {CFG.model_path}")
    print("Next: run `python -m src.evaluate` to compute metrics.")

if __name__ == "__main__":
    main()

