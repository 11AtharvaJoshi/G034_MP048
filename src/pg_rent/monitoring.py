import logging
import json
import numpy as np
import pandas as pd

FEATURES = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps"
]


def calculate_psi(reference, current, bins=10):
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref = pd.cut(reference, edges).value_counts(sort=False)
    cur = pd.cut(current, edges).value_counts(sort=False)

    ref_pct = np.clip(ref / len(reference), 1e-6, None)
    cur_pct = np.clip(cur / len(current), 1e-6, None)

    return float(((cur_pct - ref_pct) *
                  np.log(cur_pct / ref_pct)).sum())


def calculate_all_psi(train, later, bins=10):
    return {
        feature: calculate_psi(train[feature], later[feature], bins=bins)
        for feature in FEATURES
    }


def calculate_mae_by_locality(data):
    data = data.copy()
    data["absolute_error"] = (
        data["rent_rs"] - data["prediction"]
    ).abs()

    return data.groupby("locality")["absolute_error"].mean().to_dict()


def get_most_drifted_feature(psi_scores):
    return max(psi_scores, key=psi_scores.get)


class JsonFormatter(logging.Formatter):

    def format(self, record):
        log_data = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        return json.dumps(log_data)


def configure_json_logging():
    logger = logging.getLogger("pg_rent")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger