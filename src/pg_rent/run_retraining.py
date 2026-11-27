from src.pg_rent.monitoring import configure_json_logging

import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from src.pg_rent.config import load_config
from src.pg_rent.model import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.pg_rent.retraining import compare_models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = configure_json_logging()

def main():
    config = load_config()

    target = config["model"]["target"]
    fraction = config["monitoring"]["retraining_fraction"]
    threshold = config["monitoring"]["promotion_improvement"]

    train_path = PROJECT_ROOT / config["data"]["cleaned_train_path"]
    later_path = PROJECT_ROOT / config["data"]["later_path"]
    original_train_path = PROJECT_ROOT / config["data"]["train_path"]

    mlflow_db = PROJECT_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # 1. Load datasets
    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)

    # 2. Sort later data chronologically using stable sorting
    later = later.sort_values("date", kind="stable").reset_index(drop=True)

    split_index = int(len(later) * fraction)

    later_train = later.iloc[:split_index].copy()
    later_test = later.iloc[split_index:].copy()
    
    logger.info(f"Original cleaned training rows: {len(train)}")
    logger.info(f"Later training rows: {len(later_train)}")
    logger.info(f"Later evaluation rows: {len(later_test)}")

    # 3. Select only permitted model features and target
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    required_columns = features + [target]

    v2_train = pd.concat(
        [
            train[required_columns],
            later_train[required_columns]
        ],
        ignore_index=True
    )

    X_v2_train = v2_train[features]
    y_v2_train = v2_train[target]

    X_test = later_test[features]
    y_test = later_test[target]

    # 4. Evaluate original V1 on the same 90 evaluation records
    v1_uri = "models:/PG_Rent_Prediction/1"
    v1_model = mlflow.sklearn.load_model(v1_uri)

    v1_predictions = v1_model.predict(X_test)
    mae_v1 = mean_absolute_error(y_test, v1_predictions)

    logger.info(f"V1 MAE: {mae_v1:.2f}")

    # 5. Clone V1 pipeline and retrain it using expanded data
    v2_model = clone(v1_model)
    v2_model.fit(X_v2_train, y_v2_train)

    v2_predictions = v2_model.predict(X_test)
    mae_v2 = mean_absolute_error(y_test, v2_predictions)

    logger.info(f"V2 MAE: {mae_v2:.2f}")

    # 6. Apply configured promotion rule
    decision = compare_models(
        mae_v1,
        mae_v2,
        threshold=threshold
    )

    improvement = (mae_v1 - mae_v2) / mae_v1

    logger.info(f"Improvement: {improvement:.2%}")
    logger.info(f"Decision: {decision.upper()}")

    # 7. Calculate SHA256 of original train.csv
    sha256 = hashlib.sha256(original_train_path.read_bytes()).hexdigest()

    # 8. Log V2 experiment to MLflow
    with mlflow.start_run(run_name="V2_Retraining") as run:
        mlflow.log_param("model_version", "v2_candidate")
        mlflow.log_param("training_rows", len(v2_train))
        mlflow.log_param("later_training_rows", len(later_train))
        mlflow.log_param("evaluation_rows", len(later_test))
        mlflow.log_param("retraining_fraction", fraction)
        mlflow.log_param("promotion_threshold", threshold)
        mlflow.log_param("data_sha256", sha256)

        mlflow.log_metric("mae_v1", mae_v1)
        mlflow.log_metric("mae_v2", mae_v2)
        mlflow.log_metric("improvement", improvement)

        model_info = mlflow.sklearn.log_model(
            sk_model=v2_model,
            name="v2_model",
            serialization_format="cloudpickle"
        )

        logger.info(f"MLflow run ID: {run.info.run_id}")

    # 9. Register and promote only if the threshold is achieved
    registered_version = None

    if decision == "promote":
        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name="PG_Rent_Prediction"
        )

        client.set_registered_model_alias(
            "PG_Rent_Prediction",
            "champion",
            str(model_version.version)
        )

        registered_version = model_version.version
        logger.info(f"New model promoted. Registry version: {registered_version}")

    else:
        logger.info("Promotion condition not met. Champion remains unchanged.")

    # 10. Save report
    report = {
        "v1_mae": float(mae_v1),
        "v2_mae": float(mae_v2),
        "improvement_percentage": float(improvement * 100),
        "promotion_threshold_percentage": float(threshold * 100),
        "decision": decision,
        "v2_training_rows": len(v2_train),
        "later_training_rows": len(later_train),
        "evaluation_rows": len(later_test),
        "evaluation_start_date": str(later_test["date"].min()),
        "evaluation_end_date": str(later_test["date"].max()),
        "mlflow_run_id": run.info.run_id,
        "registered_version": registered_version
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "retraining_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info(f"Retraining report saved to: {report_path}")


if __name__ == "__main__":
    main()
