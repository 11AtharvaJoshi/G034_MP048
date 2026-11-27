
"""Run data drift and model performance monitoring for MP048."""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from src.pg_rent.config import load_config
from src.pg_rent.monitoring import (
    calculate_all_psi,
    calculate_mae_by_locality,
    get_most_drifted_feature,
    configure_json_logging
)

from src.pg_rent.model import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def run_monitoring():
    """Calculate PSI and evaluate v1 model on later dataset."""

    # Step 1: Configure project paths
    project_root = Path(__file__).resolve().parents[2]

    config = load_config()

    train_path = project_root / config["data"]["cleaned_train_path"]
    later_path = project_root / config["data"]["later_path"]

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Configure MLflow
    tracking_uri = f"sqlite:///{(project_root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    # Step 3: Configure logging
    logger = configure_json_logging()

    logger.info("Starting MP048 monitoring")

    # Step 4: Load datasets
    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path)

    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Later data shape: {later_df.shape}")

    # Step 5: Calculate PSI
    psi_bins = config["monitoring"]["psi_bins"]

    psi_scores = calculate_all_psi(
       train_df,
       later_df,
       bins=psi_bins
    )

    most_drifted = get_most_drifted_feature(psi_scores)

    logger.info("PSI Results:")
    
    for feature, score in psi_scores.items():
        logger.info(f"PSI for {feature}: {score:.6f}")
        
        logger.info(f"Most drifted feature: {most_drifted}")

    # Step 6: Load original v1 model
    model_uri = "models:/PG_Rent_Prediction/1"

    model = mlflow.sklearn.load_model(model_uri)

    # Step 7: Prepare later dataset features
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_later = later_df[feature_columns]
    y_later = later_df["rent_rs"]

    # Step 8: Generate v1 predictions on later data
    predictions = model.predict(X_later)

    performance_df = later_df[["locality", "rent_rs"]].copy()
    performance_df["prediction"] = predictions

    # Step 9: Calculate MAE by locality
    mae_by_locality = calculate_mae_by_locality(performance_df)

    logger.info("V1 MAE by locality:")
    for locality, mae in mae_by_locality.items():
        logger.info(f"{locality}: {mae:.2f}")

    overall_mae = (y_later - predictions).__abs__().mean()

    logger.info(f"Overall V1 MAE: {overall_mae:.2f}")

    # Step 10: Save monitoring report
    monitoring_report = {
        "psi_scores": psi_scores,
        "most_drifted_feature": most_drifted,
        "overall_v1_mae": float(overall_mae),
        "mae_by_locality": {
            locality: float(mae)
            for locality, mae in mae_by_locality.items()
        }
    }

    report_path = reports_dir / "monitoring_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(monitoring_report, file, indent=4)

    logger.info("Monitoring completed successfully")

    logger.info(f"Monitoring report saved to: {report_path}")


if __name__ == "__main__":
    run_monitoring()
