"""Model training and MLflow tracking for MP048."""

import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pg_rent.config import load_config

NUMERIC_FEATURES = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps"
]

CATEGORICAL_FEATURES = [
    "furnishing",
    "locality"
]

TARGET = "rent_rs"


def setup_mlflow():
    """Configure MLflow tracking."""

    project_root = Path(__file__).resolve().parents[2]

    tracking_uri = f"sqlite:///{(project_root / 'mlflow.db').as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("MP048_PG_Rent_Prediction")


def load_training_data():
    """Load cleaned training data and original dataset hash."""

    config = load_config()

    project_root = Path(__file__).resolve().parents[2]

    cleaned_path = project_root / config["data"]["cleaned_train_path"]
    original_path = project_root / config["data"]["train_path"]

    df = pd.read_csv(cleaned_path)

    with open(original_path, "rb") as file:
        data_sha256 = hashlib.sha256(file.read()).hexdigest()

    return df, data_sha256


def build_preprocessor():
    """Build preprocessing pipelines for numerical and categorical features."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor



def get_models(random_state=42):
    """Return candidate models with different hyperparameter settings."""

    models = {
        "LinearRegression_Default": LinearRegression(),

        "RandomForest_100": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1
        ),

        "RandomForest_300": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        ),

        "HistGradientBoosting_100": HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=random_state
        ),

        "HistGradientBoosting_200": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=15,
            random_state=random_state
        )
    }

    return models


def train_and_evaluate(X_train, X_test, y_train, y_test, data_sha256):
    """Train candidate models, evaluate MAE, and track experiments in MLflow."""

    setup_mlflow()

    config = load_config()
    random_state = config["model"]["random_state"]

    results = {}
    trained_models = {}

    models = get_models(random_state)

    for model_name, model in models.items():

        preprocessor = build_preprocessor()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        with mlflow.start_run(run_name=model_name):

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("data_sha256", data_sha256)

            mlflow.log_params(model.get_params())

            mlflow.log_metric("mae", mae)

            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=X_train.head(2),
                serialization_format="cloudpickle"
            )

            results[model_name] = mae
            trained_models[model_name] = pipeline

            print(f"{model_name} MAE: {mae:.2f}")

    best_model_name = min(results, key=results.get)

    best_model = trained_models[best_model_name]

    print("\nBest Model:", best_model_name)
    print("Best MAE:", round(results[best_model_name], 2))

    return best_model, best_model_name, results


def split_dataset(df):
    """Separate features and target, then split into training and testing sets."""

    config = load_config()

    target = config["model"]["target"]
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def register_best_model(best_model, best_model_name, best_mae, data_sha256):
    """Register the winning model and assign the champion alias."""

    registered_name = "PG_Rent_Prediction"

    setup_mlflow()
    client = MlflowClient()

    with mlflow.start_run(run_name="Best_Model_Registration") as run:

        mlflow.log_param("model_name", best_model_name)
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_metric("mae", best_mae)

        mlflow.sklearn.log_model(
            best_model,
            name="best_model",
            input_example=pd.DataFrame(
                [[1.0, 200, 2, 2, 5, 50, "semi", "Sector West"]],
                columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES
            ),
            serialization_format="cloudpickle"
        )

        model_uri = f"runs:/{run.info.run_id}/best_model"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_name
    )

    client.set_registered_model_alias(
        registered_name,
        "champion",
        str(model_version.version)
    )

    print("\nRegistered Model:", registered_name)
    print("Model Version:", model_version.version)
    print("Champion Alias: champion")

    return registered_name, model_version.version


def main():
    """Run model experiments without registering a new model."""

    print("Starting MP048 model experiments...")

    # Step 1: Load cleaned training data
    df, data_sha256 = load_training_data()

    print("Dataset loaded:", df.shape)

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = split_dataset(df)

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Step 3: Train and evaluate candidate models
    best_model, best_model_name, results = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        data_sha256
    )

    print("\nBest Model:", best_model_name)
    print("Best MAE:", round(results[best_model_name], 2))

    print("\nExperiments completed successfully!")


if __name__ == "__main__":
    main()