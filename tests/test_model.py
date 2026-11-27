"""Model pipeline and model/API integration tests for MP048."""

import math

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from app.main import app
from src.pg_rent.model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    get_models,
)


def _make_synthetic_dataset(n_rows: int = 40) -> pd.DataFrame:
    """Create a deterministic synthetic PG rent dataset without external CSVs."""
    rng = np.random.default_rng(42)
    furnishings = np.array(["unfurnished", "semi", "full"])
    localities = np.array(["Sector East", "Sector West", "Market Side"])

    distance = rng.uniform(0.5, 10.0, size=n_rows)
    area = rng.uniform(100.0, 400.0, size=n_rows)
    sharing = rng.integers(1, 5, size=n_rows)
    meals = rng.integers(0, 4, size=n_rows)
    age = rng.uniform(1.0, 40.0, size=n_rows)
    wifi = rng.uniform(10.0, 300.0, size=n_rows)
    furnishing = rng.choice(furnishings, size=n_rows)
    locality = rng.choice(localities, size=n_rows)

    furnishing_bonus = np.where(furnishing == "full", 3000.0, np.where(furnishing == "semi", 1500.0, 0.0))
    rent = (
        9000.0
        + 35.0 * area
        - 250.0 * distance
        - 1200.0 * sharing
        + 900.0 * meals
        - 40.0 * age
        + 5.0 * wifi
        + furnishing_bonus
    )

    return pd.DataFrame(
        {
            "distance_to_campus_km": distance,
            "room_area_sqft": area,
            "sharing": sharing,
            "meals_per_day": meals,
            "building_age_yr": age,
            "wifi_mbps": wifi,
            "furnishing": furnishing,
            "locality": locality,
            "rent_rs": rent,
        }
    )


def test_get_models_provides_multiple_candidates():
    models = get_models(random_state=42)
    assert len(models) >= 3
    assert "LinearRegression_Default" in models


def test_pipeline_trains_and_achieves_low_mae_on_synthetic_data():
    df = _make_synthetic_dataset(n_rows=60)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    train_df = df.iloc[:45]
    eval_df = df.iloc[45:]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", get_models(random_state=42)["LinearRegression_Default"]),
        ]
    )
    pipeline.fit(train_df[feature_columns], train_df["rent_rs"])
    preds = pipeline.predict(eval_df[feature_columns])

    assert len(preds) == len(eval_df)
    assert all(math.isfinite(float(val)) for val in preds)
    assert mean_absolute_error(eval_df["rent_rs"], preds) < 1000.0


def test_pipeline_handles_missing_numeric_and_unseen_categories():
    df = _make_synthetic_dataset(n_rows=30)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", get_models(random_state=42)["LinearRegression_Default"]),
        ]
    )
    pipeline.fit(df[feature_columns], df["rent_rs"])

    edge_input = pd.DataFrame(
        [
            {
                "distance_to_campus_km": np.nan,
                "room_area_sqft": 210.0,
                "sharing": 2,
                "meals_per_day": 2,
                "building_age_yr": 12.0,
                "wifi_mbps": 100.0,
                "furnishing": "luxury_unknown",
                "locality": "New Campus Road",
            }
        ]
    )[feature_columns]

    pred = pipeline.predict(edge_input)
    assert len(pred) == 1
    assert math.isfinite(float(pred[0]))


def test_api_prediction_matches_fitted_model_output():
    df = _make_synthetic_dataset(n_rows=40)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", get_models(random_state=42)["LinearRegression_Default"]),
        ]
    )
    pipeline.fit(df[feature_columns], df["rent_rs"])

    payload = {
        "distance_to_campus_km": 3.2,
        "room_area_sqft": 240.0,
        "sharing": 2,
        "meals_per_day": 2,
        "building_age_yr": 8.0,
        "wifi_mbps": 120.0,
        "furnishing": "full",
        "locality": "Sector East",
    }
    expected_prediction = round(float(pipeline.predict(pd.DataFrame([payload])[feature_columns])[0]), 2)

    previous_model = getattr(app.state, "model", None)
    app.state.model = pipeline
    try:
        client = TestClient(app)
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["prediction"] == expected_prediction
    finally:
        app.state.model = previous_model
