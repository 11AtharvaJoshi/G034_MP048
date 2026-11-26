
"""Generate holdout predictions using the registered champion model."""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from src.pg_rent.config import load_config
from src.pg_rent.model import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def predict_holdout():
    """Load the champion model and generate holdout predictions."""

    # Step 1: Configure paths and MLflow
    project_root = Path(__file__).resolve().parents[2]

    tracking_uri = f"sqlite:///{(project_root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    config = load_config()

    holdout_path = project_root / config["data"]["holdout_path"]
    output_path = project_root / "data/processed/holdout_predictions.csv"

    # Step 2: Load holdout dataset
    holdout_df = pd.read_csv(holdout_path)

    print("Holdout dataset shape:", holdout_df.shape)

    # Step 3: Select only the eight model features
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_holdout = holdout_df[feature_columns]

    # Step 4: Load registered champion model
    model_uri = "models:/PG_Rent_Prediction@champion"

    model = mlflow.sklearn.load_model(model_uri)

    print("Loaded registered champion model.")

    # Step 5: Generate predictions
    predictions = model.predict(X_holdout)

    # Step 6: Create required output format
    output_df = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": predictions
    })

    # Step 7: Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(output_path, index=False)

    print("\nPredictions generated successfully!")
    print("Total predictions:", len(output_df))
    print("Output file:", output_path)

    print("\nFirst 5 predictions:")
    print(output_df.head().to_string(index=False))


if __name__ == "__main__":
    predict_holdout()