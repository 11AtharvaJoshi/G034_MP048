"""API tests for the MP048 FastAPI prediction service."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app.main import RentPredictionRequest, app
from src.pg_rent.model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor


VALID_PAYLOAD = {
    "distance_to_campus_km": 2.5,
    "room_area_sqft": 200.0,
    "sharing": 2,
    "meals_per_day": 2,
    "building_age_yr": 10.0,
    "wifi_mbps": 50.0,
    "furnishing": "semi",
    "locality": "Sector West",
}


def _build_synthetic_pipeline() -> Pipeline:
    """Fit a small deterministic pipeline on in-memory synthetic data."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    synthetic_df = pd.DataFrame(
        [
            [1.0, 150.0, 2, 1, 5.0, 50.0, "unfurnished", "Sector East", 11000.0],
            [3.0, 220.0, 2, 2, 10.0, 100.0, "semi", "Sector West", 15500.0],
            [5.0, 300.0, 1, 3, 2.0, 200.0, "full", "Market Side", 24000.0],
            [2.0, 180.0, 3, 2, 15.0, 40.0, "semi", "Sector East", 13200.0],
            [6.0, 260.0, 2, 1, 25.0, 80.0, "full", "Sector West", 17800.0],
            [4.0, 120.0, 4, 0, 30.0, 20.0, "unfurnished", "Market Side", 9500.0],
        ],
        columns=feature_columns + ["rent_rs"],
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(synthetic_df[feature_columns], synthetic_df["rent_rs"])
    return pipeline


@pytest.fixture
def client():
    """Provide a TestClient backed by an in-memory fitted model."""
    previous_model = getattr(app.state, "model", None)
    app.state.model = _build_synthetic_pipeline()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.state.model = previous_model


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_request_schema_has_exact_eight_fields():
    expected_fields = {
        "distance_to_campus_km",
        "room_area_sqft",
        "sharing",
        "meals_per_day",
        "building_age_yr",
        "wifi_mbps",
        "furnishing",
        "locality",
    }
    assert set(RentPredictionRequest.model_fields.keys()) == expected_fields


def test_predict_valid_payload(client: TestClient, capsys):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)
    assert body["prediction"] > 0

    captured = capsys.readouterr()
    log_lines = [line for line in captured.out.strip().splitlines() if line.strip()]
    assert log_lines, "Expected JSON log line on stdout"
    parsed_log = json.loads(log_lines[-1])
    assert parsed_log["event"] == "predict"
    assert parsed_log["prediction"] == body["prediction"]


@pytest.mark.parametrize("invalid_age", [75.1, 76, 120])
def test_predict_rejects_building_age_above_75(client: TestClient, invalid_age):
    payload = {**VALID_PAYLOAD, "building_age_yr": invalid_age}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_accepts_building_age_at_boundary_75(client: TestClient):
    payload = {**VALID_PAYLOAD, "building_age_yr": 75}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


@pytest.mark.parametrize("invalid_wifi", [-0.1, -1, -50])
def test_predict_rejects_negative_wifi_mbps(client: TestClient, invalid_wifi):
    payload = {**VALID_PAYLOAD, "wifi_mbps": invalid_wifi}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_accepts_wifi_mbps_at_boundary_zero(client: TestClient):
    payload = {**VALID_PAYLOAD, "wifi_mbps": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_rejects_missing_field(client: TestClient):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "locality"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
