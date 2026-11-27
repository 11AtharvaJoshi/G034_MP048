
import pandas as pd

from src.pg_rent.cleaning import clean_training_data


def sample_data():
    return pd.DataFrame({
        "record_id": ["R1", "R1", "R2"],
        "wifi_mbps": [20, 20, -1],
        "rent_in_signed_agreement": [15000, 15000, 16000],
        "rent_rs": [15000, 15000, 16000],
        "furnishing": ["unf.", "unf.", "semi"]
    })


def test_clean_training_data_removes_duplicates():
    cleaned = clean_training_data(sample_data())

    assert len(cleaned) == 2
    assert cleaned["record_id"].is_unique


def test_clean_training_data_removes_leakage_column():
    cleaned = clean_training_data(sample_data())

    assert "rent_in_signed_agreement" not in cleaned.columns


def test_clean_training_data_handles_sentinel_and_furnishing():
    cleaned = clean_training_data(sample_data())

    assert (cleaned["wifi_mbps"] >= 0).all()
    assert cleaned.iloc[1]["wifi_mbps"] == 20
    assert cleaned.iloc[0]["furnishing"] == "unfurnished"