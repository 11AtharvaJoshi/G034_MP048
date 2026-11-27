"""FastAPI prediction service for MP048 (PG Room Rent - Indore)."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pg_rent.config import load_config

NUMERIC_FEATURES = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps",
]

CATEGORICAL_FEATURES = [
    "furnishing",
    "locality",
]

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON written to stdout."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("event", "prediction", "furnishing", "locality"):
            if hasattr(record, field):
                log_payload[field] = getattr(record, field)
        return json.dumps(log_payload)


class StdoutStreamHandler(logging.StreamHandler):
    """StreamHandler that always writes to the active sys.stdout stream."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def configure_logger() -> logging.Logger:
    """Configure structured JSON logging to stdout without duplicating handlers."""
    service_logger = logging.getLogger("pg_rent.api")
    service_logger.setLevel(logging.INFO)
    service_logger.propagate = False

    if not service_logger.handlers:
        handler = StdoutStreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        service_logger.addHandler(handler)

    return service_logger


logger = configure_logger()


class RentPredictionRequest(BaseModel):
    """Request body schema for POST /predict with client validation rules."""

    distance_to_campus_km: float
    room_area_sqft: float
    sharing: int
    meals_per_day: int
    building_age_yr: float = Field(
        ...,
        le=75,
        description="Age of the building in years; values above 75 are rejected.",
    )
    wifi_mbps: float = Field(
        ...,
        ge=0,
        description="Wi-Fi speed in Mbps; negative values are rejected.",
    )
    furnishing: str
    locality: str


class RentPredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    prediction: float


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    model_loaded: bool


def _build_fallback_pipeline() -> Pipeline:
    """Build the winning preprocessing + LinearRegression pipeline from MP048."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def load_model():
    """Load the champion model from joblib, MLflow registry, or cleaned training data."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    mlflow_db = PROJECT_ROOT / "mlflow.db"
    if mlflow_db.exists() and (PROJECT_ROOT / "mlruns").exists():
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
            return mlflow.sklearn.load_model("models:/PG_Rent_Prediction@champion")
        except Exception as exc:
            logger.warning("MLflow champion unavailable locally; falling back: %s", exc)

    config = load_config()
    cleaned_path = PROJECT_ROOT / config["data"]["cleaned_train_path"]
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        pipeline = _build_fallback_pipeline()
        pipeline.fit(df[feature_columns], df[config["model"]["target"]])
        return pipeline

    raise RuntimeError(
        "Could not load model: neither models/model.joblib, local MLflow artifacts, "
        "nor cleaned training data were found."
    )


def get_model(fastapi_app: FastAPI):
    """Return the cached model on app.state, loading it on first use if needed."""
    model = getattr(fastapi_app.state, "model", None)
    if model is None:
        model = load_model()
        fastapi_app.state.model = model
    return model


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Warm-load the prediction model when the FastAPI app starts."""
    if getattr(fastapi_app.state, "model", None) is None:
        try:
            fastapi_app.state.model = load_model()
        except Exception as exc:
            logger.warning("Model warm-load skipped at startup: %s", exc)
    yield


app = FastAPI(
    title="MP048 PG Room Rent Prediction API",
    description="Predicts fair monthly PG room rent in Indore.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint for readiness and liveness probes."""
    model = getattr(app.state, "model", None)
    if model is None:
        try:
            model = get_model(app)
        except Exception:
            model = None
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/predict", response_model=RentPredictionResponse)
def predict(payload: RentPredictionRequest) -> RentPredictionResponse:
    """Predict monthly PG rent in rupees from validated room features."""
    model = get_model(app)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    input_df = pd.DataFrame([payload.model_dump()])[feature_columns]

    predicted_rent = float(model.predict(input_df)[0])

    logger.info(
        "Prediction generated",
        extra={
            "event": "predict",
            "prediction": round(predicted_rent, 2),
            "furnishing": payload.furnishing,
            "locality": payload.locality,
        },
    )

    return RentPredictionResponse(prediction=round(predicted_rent, 2))
