
import pandas as pd
from pathlib import Path

TRAIN_PATH = Path("data/raw/train.csv")

NUMERIC_FEATURES = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps",
]


def investigate_unit_mixup(df):
    print("\n========== UNIT MIX-UP INVESTIGATION ==========")

    for column in NUMERIC_FEATURES:
        print(f"\nFeature: {column}")
        summary = df.groupby("locality")[column].agg(
            ["min", "mean", "max"]
        )
        print(summary.to_string())



def investigate_stuck_sensor(df):
    print("\n========== STUCK SENSOR INVESTIGATION ==========")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["date", "record_id"]).reset_index(drop=True)

    for column in NUMERIC_FEATURES:
        values = df[column]

        runs = []
        start = 0

        for i in range(1, len(df) + 1):
            if (
                i == len(df)
                or pd.isna(values.iloc[i])
                or pd.isna(values.iloc[i - 1])
                or values.iloc[i] != values.iloc[i - 1]
            ):
                if not pd.isna(values.iloc[start]):
                    run = df.iloc[start:i]

                    runs.append({
                        "value": values.iloc[start],
                        "rows": len(run),
                        "start_date": run["date"].iloc[0],
                        "end_date": run["date"].iloc[-1],
                        "duration_days": (
                            run["date"].iloc[-1] -
                            run["date"].iloc[0]
                        ).days
                    })

                start = i

        runs.sort(
            key=lambda x: (x["duration_days"], x["rows"]),
            reverse=True
        )

        print(f"\nFeature: {column}")

        for run in runs[:3]:
            print(
                f"Value={run['value']}, "
                f"Rows={run['rows']}, "
                f"Start={run['start_date'].date()}, "
                f"End={run['end_date'].date()}, "
                f"Duration={run['duration_days']} days"
            )

def main():
    df = pd.read_csv(TRAIN_PATH)

    print("Training dataset shape:", df.shape)
    print("Unique localities:", df["locality"].unique())

    investigate_unit_mixup(df)
    investigate_stuck_sensor(df)


if __name__ == "__main__":
    main()
