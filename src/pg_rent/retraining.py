import logging
from src.pg_rent.monitoring import configure_json_logging

logger = configure_json_logging()


def compare_models(mae_v1, mae_v2, threshold=0.10):

    if mae_v1 <= 0 or mae_v2 < 0:
        raise ValueError("Invalid MAE values")

    improvement = (mae_v1 - mae_v2) / mae_v1

    if improvement >= threshold:
        decision = "promote"
    else:
        decision = "keep"

    logger.info(
        f"V1 MAE: {mae_v1}, V2 MAE: {mae_v2}, "
        f"Improvement: {improvement:.2%}, Decision: {decision}"
    )

    return decision


def should_promote(mae_v1, mae_v2):
    return compare_models(mae_v1, mae_v2) == "promote"