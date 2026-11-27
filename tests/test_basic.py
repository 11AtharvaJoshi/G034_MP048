
import csv
from pathlib import Path

import pandas as pd

from src.pg_rent.cleaning import clean_training_data


def test_train_dataset_exists():
    file_path = Path("data/raw/train.csv")
    assert file_path.exists()


def test_train_dataset_has_required_columns():
    file_path = Path("data/raw/train.csv")

    with file_path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file)
        columns = next(reader)

    required_columns = [
        "record_id",
        "date",
        "distance_to_campus_km",
        "room_area_sqft",
        "sharing",
        "meals_per_day",
        "building_age_yr",
        "wifi_mbps",
        "furnishing",
        "locality",
        "rent_in_signed_agreement",
        "rent_rs"
    ]

    assert all(column in columns for column in required_columns)


def test_clean_training_data_removes_duplicates_leakage_and_sentinel():
    data = pd.DataFrame({
        "record_id": ["R1", "R1", "R2"],
        "wifi_mbps": [20, 20, -1],
        "rent_in_signed_agreement": [15000, 15000, 16000],
        "rent_rs": [15000, 15000, 16000],
        "furnishing": ["unf.", "unf.", "semi"]
    })

    cleaned = clean_training_data(data)

    assert len(cleaned) == 2
    assert "rent_in_signed_agreement" not in cleaned.columns
    assert (cleaned["wifi_mbps"] >= 0).all()
    assert cleaned.loc[1, "wifi_mbps"] == 20
    assert cleaned.loc[0, "furnishing"] == "unfurnished"
