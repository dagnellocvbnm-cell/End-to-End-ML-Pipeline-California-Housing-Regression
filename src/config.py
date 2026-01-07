from dataclasses import dataclass

#marks class as a dataclass, instances are immutable
@dataclass(frozen=True)
class Config:
    random_seed: int = 42

    #20% test, 80% train
    test_size: float = 0.2

    # Ridge is a strong default baseline for numeric regression.  Alpha determines regularization strength.
    ridge_alpha: float = 1.0

    #Stores DummyRegressor, trained Pipeline and feature metadata
    model_path: str = "models/model.joblib"
    
    #To store MAE, RMSE and R^2 for each model
    metrics_path: str = "reports/metrics.json"

CFG = Config()

