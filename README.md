# End-to-End ML Pipeline (California Housing Regression)

This project builds a reproducible regression ML pipeline in Python using scikit-learn to predict median house value from demographic and geographic features.

## Pipeline Stages
1. Ingest: load the California Housing dataset (sklearn)
2. Split: reproducible train/test split (fixed random seed)
3. Preprocess: median imputation + standard scaling (leakage-safe Pipeline)
4. Train: baseline models (DummyRegressor vs Ridge regression)
5. Evaluate: MAE, RMSE, R²
6. Artifacts: generate trained model artifacts (.joblib) and evaluation metrics (.json) locally

## How to Run
```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
```
