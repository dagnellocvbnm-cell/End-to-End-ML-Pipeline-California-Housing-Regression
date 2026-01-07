from __future__ import annotations

import json
import os
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import CFG
from src.data import load_and_split

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics.

    Variables:
      - y_true: true targets (shape: n)
      - y_pred: predicted targets (shape: n)
      - n: number of examples
      - MAE: mean(|y_true - y_pred|)
      - RMSE: sqrt(mean((y_true - y_pred)^2))
      - R2: 1 - SS_res/SS_tot
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}

def main() -> None:
    split = load_and_split(test_size=CFG.test_size, random_seed=CFG.random_seed)

    bundle: Dict[str, Any] = joblib.load(CFG.model_path)
    dummy = bundle["dummy"]
    pipeline = bundle["pipeline"]

    y_pred_dummy = dummy.predict(split.X_test)
    y_pred_pipe = pipeline.predict(split.X_test)

    metrics = {
        "dataset": "sklearn.fetch_california_housing",
        "split": {"test_size": CFG.test_size, "random_seed": CFG.random_seed},
        "models": {
            "dummy_mean": regression_metrics(split.y_test, y_pred_dummy),
            "ridge_pipeline": regression_metrics(split.y_test, y_pred_pipe),
        },
    }

    os.makedirs(os.path.dirname(CFG.metrics_path), exist_ok=True)
    with open(CFG.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation complete.")
    print(json.dumps(metrics["models"], indent=2))
    print(f"Saved metrics to: {CFG.metrics_path}")

if __name__ == "__main__":
    main()

