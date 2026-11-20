
from pathlib import Path
import pandas as pd

from src.pg_rent.config import load_config


def run_data_audit():
    config = load_config()

    train_path = config["data"]["train_path"]
    later_path = config["data"]["later_path"]

    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)

    print("\n========== DATA AUDIT ==========")

    # 1. Missing values
    print("\n1. Missing values:")
    print(train.isnull().sum())

    # 2. Duplicate records
    print("\n2. Duplicate rows:")
    print(train.duplicated().sum())

    print("Duplicate record IDs:")
    print(train["record_id"].duplicated().sum())

    # 3. Sentinel values
    print("\n3. Sentinel values:")
    print(train["wifi_mbps"].value_counts().get(-1, 0))

    # 4. Possible target typos
    print("\n4. Possible target typos:")
    ratio = (
        train["rent_rs"] /
        train["rent_in_signed_agreement"]
    )
    suspicious = train[(ratio >= 9) & (ratio <= 11)]
    print("Suspicious rows:", len(suspicious))
    print("Unique IDs:", suspicious["record_id"].nunique())

    # 5. Spelling variants
    print("\n5. Furnishing categories:")
    print(train["furnishing"].value_counts())

    print("\nLater dataset categories:")
    print(later["furnishing"].value_counts())

    # 6. Possible target leakage
    print("\n6. Feature availability:")
    print(
        "Agreement rent in train:",
        "rent_in_signed_agreement" in train.columns
    )
    print(
        "Agreement rent in holdout:",
        "rent_in_signed_agreement" in pd.read_csv(
            config["data"]["holdout_path"], nrows=1
        ).columns
    )

    # 7. Stuck sensor / unit mix-up
    print("\n7. Stuck sensor and unit mix-up:")
    print(
        "These require date-run and group-wise checks; "
        "see the investigation findings before marking them."
    )

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    run_data_audit()
