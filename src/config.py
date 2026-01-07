from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    random_seed: int = 42
    test_size: float = 0.2

    # Ridge is a strong default baseline for numeric regression.
    ridge_alpha: float = 1.0

    model_path: str = "models/model.joblib"
    metrics_path: str = "reports/metrics.json"

CFG = Config()

