
import csv
from pathlib import Path


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
