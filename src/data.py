from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: Tuple[str, ...]
    target_name: str

def load_and_split(*, test_size: float, random_seed: int) -> DatasetSplit:
    """
    Loads the sklearn California Housing dataset and splits into train/test.

    Variables:
      - X: feature matrix (n_samples x n_features)
      - y: target vector (n_samples,)
    """
    bunch = fetch_california_housing(as_frame=False)
    X = bunch.data
    y = bunch.target
    feature_names = tuple(bunch.feature_names)
    target_name = "MedHouseVal"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_name=target_name,
    )

