
import pandas as pd

from src.pg_rent.config import load_config


NUMERIC_FEATURES = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps",
]


def investigate_unit_mixup(train):
    print("\n7A. UNIT MIX-UP INVESTIGATION")

    for column in NUMERIC_FEATURES:
        print(f"\nFeature: {column}")

        summary = train.groupby("locality")[column].agg(
            ["min", "mean", "max"]
        )

        print(summary.to_string())


def investigate_stuck_sensor(train):
    print("\n7B. STUCK SENSOR INVESTIGATION")

    data = train.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    data = data.sort_values(["date", "record_id"]).reset_index(drop=True)

    for column in NUMERIC_FEATURES:
        values = data[column].reset_index(drop=True)

        longest_run = None
        current_start = 0

        for i in range(1, len(values) + 1):
            run_ended = (
                i == len(values)
                or pd.isna(values.iloc[i])
                or pd.isna(values.iloc[i - 1])
                or values.iloc[i] != values.iloc[i - 1]
            )

            if run_ended:
                if not pd.isna(values.iloc[current_start]):
                    run = data.iloc[current_start:i]

                    candidate = {
                        "value": values.iloc[current_start],
                        "rows": len(run),
                        "start_date": run["date"].iloc[0],
                        "end_date": run["date"].iloc[-1],
                        "duration_days": (
                            run["date"].iloc[-1] - run["date"].iloc[0]
                        ).days,
                    }

                    if (
                        longest_run is None
                        or candidate["rows"] > longest_run["rows"]
                    ):
                        longest_run = candidate

                current_start = i

        print(f"\nFeature: {column}")

        if longest_run is None:
            print("No valid repeated-value run found.")
        else:
            print("Repeated value:", longest_run["value"])
            print("Rows in longest consecutive run:", longest_run["rows"])
            print("Start date:", longest_run["start_date"].date())
            print("End date:", longest_run["end_date"].date())
            print("Duration in days:", longest_run["duration_days"])


def run_data_audit():
    config = load_config()

    train_path = config["data"]["train_path"]
    later_path = config["data"]["later_path"]
    holdout_path = config["data"]["holdout_path"]

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

    holdout = pd.read_csv(holdout_path, nrows=1)

    print(
        "Agreement rent in holdout:",
        "rent_in_signed_agreement" in holdout.columns
    )

    # 7. Unit mix-up and stuck sensor
    investigate_unit_mixup(train)
    investigate_stuck_sensor(train)

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    run_data_audit()
