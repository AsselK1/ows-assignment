from importlib import import_module
from collections.abc import Callable
from typing import cast

from src.analytics.anomaly_price import detect_price_anomalies
from src.analytics.queries import (
    spend_by_enstr,
    spend_by_region,
    supplier_concentration,
    total_spend_by_bin,
    year_over_year_trends,
)
from src.analytics.statistics import (
    clickhouse_median,
    clickhouse_percentile,
    clickhouse_quantiles,
    confidence_interval,
    iqr,
    mad,
    median,
    percentile,
    percentiles,
    weighted_mean,
)


def detect_fairness_anomalies(*args: object, **kwargs: object) -> dict[str, int]:
    module = import_module("src.analytics.anomaly_fairness")
    detector = cast(
        "Callable[..., dict[str, int]]",
        getattr(module, "detect_fairness_anomalies"),
    )
    return detector(*args, **kwargs)


def detect_volume_anomalies(*args: object, **kwargs: object) -> dict[str, int]:
    module = import_module("src.analytics.anomaly_volume")
    detector = cast(
        "Callable[..., dict[str, int]]",
        getattr(module, "detect_volume_anomalies"),
    )
    return detector(*args, **kwargs)


__all__ = [
    "detect_price_anomalies",
    "detect_volume_anomalies",
    "detect_fairness_anomalies",
    "total_spend_by_bin",
    "spend_by_enstr",
    "spend_by_region",
    "supplier_concentration",
    "year_over_year_trends",
    "median",
    "weighted_mean",
    "iqr",
    "mad",
    "percentile",
    "percentiles",
    "confidence_interval",
    "clickhouse_median",
    "clickhouse_percentile",
    "clickhouse_quantiles",
]
