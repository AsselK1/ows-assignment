from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Protocol, cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.analytics.statistics import weighted_mean

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")
BASELINE_DATE = datetime(2024, 1, 1)
BASE_THRESHOLD_PCT = Decimal("30.00")
DEFAULT_MIN_SAMPLE_SIZE = 5
DEFAULT_INFLATION_RATE = Decimal("0.03")
URBAN_KATO_PREFIXES = frozenset({"71", "75", "79"})


class ClickHouseClient(Protocol):
    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]: ...

    def disconnect(self) -> None: ...


def _load_yaml(path: Path) -> object:
    yaml_module = importlib.import_module("yaml")
    safe_load = cast(Callable[[str], object], getattr(yaml_module, "safe_load"))
    return safe_load(path.read_text(encoding="utf-8"))


def _as_object_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return None


def _load_config(config_path: Path) -> Mapping[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    loaded_obj = _load_yaml(config_path)
    loaded = _as_object_mapping(loaded_obj)
    if loaded is None:
        raise ValueError("config.yaml must be a dictionary at top level")
    return loaded


def _to_uint64(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))

    text = str(value).strip()
    if not text:
        return default
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return default
    try:
        return max(0, int(digits))
    except ValueError:
        return default


def _to_decimal(
    value: object,
    *,
    scale: int = 2,
    default: Decimal | None = None,
) -> Decimal:
    fallback = default if default is not None else Decimal("0.00")
    if value is None:
        return fallback

    parsed: Decimal
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, int):
        parsed = Decimal(value)
    elif isinstance(value, float):
        parsed = Decimal(str(value))
    else:
        text = str(value).strip().replace(" ", "")
        if not text:
            return fallback
        if text.count(",") == 1 and text.count(".") == 0:
            text = text.replace(",", ".")
        try:
            parsed = Decimal(text)
        except Exception:  # noqa: BLE001
            return fallback

    quant = Decimal("1") if scale <= 0 else Decimal(f"1.{'0' * scale}")
    return parsed.quantize(quant, rounding=ROUND_HALF_UP)


def _parse_datetime(value: object, *, default: datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    if value is None:
        return default

    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=UTC).replace(tzinfo=None)

    raw = str(value).strip()
    if not raw:
        return default
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(UTC).replace(tzinfo=None)
    except ValueError:
        pass

    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, pattern)
        except ValueError:
            continue
    return default


def _kato_adjustment(kato_code: str | None) -> Decimal:
    if not kato_code:
        return Decimal("0.00")
    normalized = "".join(ch for ch in kato_code if ch.isdigit())
    if len(normalized) < 2:
        return Decimal("0.00")
    prefix = normalized[:2]
    if prefix in URBAN_KATO_PREFIXES:
        return Decimal("0.10")
    if prefix.isdigit():
        return Decimal("-0.05")
    return Decimal("0.00")


def _inflation_factor(sign_date: datetime, *, annual_rate: Decimal) -> Decimal:
    if sign_date >= BASELINE_DATE:
        return Decimal("1.0000")
    years_delta = max(0, BASELINE_DATE.year - sign_date.year)
    if years_delta == 0:
        return Decimal("1.0000")
    return (Decimal("1.00") + annual_rate) ** years_delta


def _classify_severity(deviation_pct: Decimal) -> str:
    if deviation_pct > Decimal("100.00"):
        return "high"
    if deviation_pct >= Decimal("50.00"):
        return "medium"
    return "low"


def _stable_uint64(parts: Sequence[object]) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _get_clickhouse_client() -> ClickHouseClient:
    host = os.getenv("CLICKHOUSE_HOST")
    if not host:
        raise ValueError("CLICKHOUSE_HOST is required")

    port_raw = os.getenv("CLICKHOUSE_PORT", "9440")
    try:
        port = int(port_raw)
    except ValueError as error:
        raise ValueError(f"CLICKHOUSE_PORT must be an integer, got: {port_raw}") from error

    database = os.getenv("CLICKHOUSE_DB")
    if not database:
        raise ValueError("CLICKHOUSE_DB is required")

    user = os.getenv("CLICKHOUSE_USER", "default")
    password = os.getenv("CLICKHOUSE_PASSWORD", "")

    is_secure = port in (9440, 8443)
    clickhouse_driver = importlib.import_module("clickhouse_driver")
    client_factory = cast(Callable[..., ClickHouseClient], getattr(clickhouse_driver, "Client"))
    return client_factory(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        secure=is_secure,
        connect_timeout=15,
        send_receive_timeout=30,
    )


def _ensure_anomaly_table(clickhouse_client: ClickHouseClient) -> None:
    _ = clickhouse_client.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_results
        (
            id UInt64,
            anomaly_type String,
            entity_type String,
            entity_id UInt64,
            detected_at DateTime,
            severity String,
            deviation_pct Decimal128(2),
            expected_value Decimal128(2),
            actual_value Decimal128(2),
            sample_size UInt32,
            enstr_code String,
            kato_code Nullable(String),
            metadata String,
            updated_at DateTime DEFAULT now(),

            INDEX idx_anomaly_type anomaly_type TYPE set(16) GRANULARITY 1,
            INDEX idx_entity_type entity_type TYPE set(16) GRANULARITY 1,
            INDEX idx_entity_id entity_id TYPE bloom_filter(0.01) GRANULARITY 2,
            INDEX idx_detected_at detected_at TYPE minmax GRANULARITY 1
        )
        ENGINE = MergeTree
        PARTITION BY toYYYYMM(detected_at)
        ORDER BY (anomaly_type, entity_type, entity_id, detected_at, id)
        """
    )


def _load_latest_contracts(clickhouse_client: ClickHouseClient) -> list[tuple[object, ...]]:
    return clickhouse_client.execute(
        """
        SELECT
            id,
            customer_bin,
            supplier_bin,
            contract_sum,
            sign_date,
            enstr_code_lvl4,
            kato_code
        FROM
        (
            SELECT
                id,
                argMax(customer_bin, updated_at) AS customer_bin,
                argMax(supplier_bin, updated_at) AS supplier_bin,
                argMax(contract_sum, updated_at) AS contract_sum,
                argMax(sign_date, updated_at) AS sign_date,
                argMax(enstr_code_lvl4, updated_at) AS enstr_code_lvl4,
                argMax(kato_code, updated_at) AS kato_code
            FROM contracts
            GROUP BY id
        )
        WHERE contract_sum > 0
        """
    )


def detect_price_anomalies(
    clickhouse_client: ClickHouseClient,
    config: Mapping[str, object],
    *,
    force: bool = False,
) -> dict[str, int]:
    _ensure_anomaly_table(clickhouse_client)
    if force:
        _ = clickhouse_client.execute("TRUNCATE TABLE anomaly_results")

    analytics_cfg = _as_object_mapping(config.get("analytics")) or {}
    min_sample_size = _to_uint64(
        analytics_cfg.get("min_sample_size"), default=DEFAULT_MIN_SAMPLE_SIZE
    )
    if min_sample_size <= 0:
        min_sample_size = DEFAULT_MIN_SAMPLE_SIZE
    inflation_rate = _to_decimal(
        analytics_cfg.get("annual_inflation_rate"),
        scale=4,
        default=DEFAULT_INFLATION_RATE,
    )

    rows = _load_latest_contracts(clickhouse_client)
    grouped: dict[str, list[tuple[int, Decimal, datetime, str | None]]] = defaultdict(list)

    now_dt = datetime.now(UTC).replace(tzinfo=None)
    for row in rows:
        if len(row) != 7:
            continue
        contract_id = _to_uint64(row[0], default=0)
        if contract_id <= 0:
            continue
        contract_sum = _to_decimal(row[3], scale=2, default=Decimal("0.00"))
        if contract_sum <= 0:
            continue
        sign_date = _parse_datetime(row[4], default=now_dt)
        enstr_raw = str(row[5] if row[5] is not None else "").strip()
        enstr_digits = "".join(ch for ch in enstr_raw if ch.isdigit())
        enstr_code_lvl4 = enstr_digits[:4]
        if len(enstr_code_lvl4) < 4:
            continue
        kato_raw = str(row[6]).strip() if row[6] is not None else None
        kato_code = kato_raw if kato_raw else None
        grouped[enstr_code_lvl4].append((contract_id, contract_sum, sign_date, kato_code))

    analyzed_categories = 0
    skipped_categories = 0
    anomalies_found = 0
    detected_at = now_dt

    insert_rows: list[tuple[object, ...]] = []
    for enstr_code, contracts in grouped.items():
        sample_size = len(contracts)
        if sample_size < min_sample_size:
            skipped_categories += 1
            LOGGER.warning(
                "Skipping ENSTR category due to small sample size: enstr=%s n=%d min_n=%d",
                enstr_code,
                sample_size,
                min_sample_size,
            )
            continue

        analyzed_categories += 1
        values = [contract_sum for _, contract_sum, _, _ in contracts]
        weights = [Decimal("1.00") for _ in contracts]
        category_baseline = weighted_mean(values, weights)

        for contract_id, actual_value, sign_date, kato_code in contracts:
            regional_adjustment = _kato_adjustment(kato_code)
            inflation_factor = _inflation_factor(sign_date, annual_rate=inflation_rate)
            expected_value = (
                category_baseline * inflation_factor * (Decimal("1.00") + regional_adjustment)
            )
            if expected_value <= 0:
                continue

            deviation_ratio = (actual_value - expected_value) / expected_value
            deviation_pct = (deviation_ratio * Decimal("100")).quantize(
                Decimal("1.00"), rounding=ROUND_HALF_UP
            )

            if deviation_pct <= BASE_THRESHOLD_PCT:
                continue

            threshold_value = (
                expected_value * (Decimal("1.00") + (BASE_THRESHOLD_PCT / Decimal("100")))
            ).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP)
            if actual_value <= threshold_value:
                continue

            severity = _classify_severity(deviation_pct)
            anomaly_id = _stable_uint64(
                ["price", "contract", contract_id, enstr_code, detected_at.isoformat()]
            )
            metadata = {
                "baseline_weighted_avg": str(category_baseline.quantize(Decimal("1.00"))),
                "regional_adjustment_pct": str(
                    (regional_adjustment * Decimal("100")).quantize(Decimal("1.00"))
                ),
                "inflation_factor": str(inflation_factor.quantize(Decimal("1.0000"))),
                "threshold_pct": str(BASE_THRESHOLD_PCT),
                "threshold_value": str(threshold_value),
                "weight_mode": "count",
            }
            insert_rows.append(
                (
                    anomaly_id,
                    "price",
                    "contract",
                    contract_id,
                    detected_at,
                    severity,
                    deviation_pct,
                    expected_value.quantize(Decimal("1.00"), rounding=ROUND_HALF_UP),
                    actual_value.quantize(Decimal("1.00"), rounding=ROUND_HALF_UP),
                    sample_size,
                    enstr_code,
                    kato_code,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    detected_at,
                )
            )
            anomalies_found += 1

    if insert_rows:
        _ = clickhouse_client.execute(
            """
            INSERT INTO anomaly_results
            (
                id,
                anomaly_type,
                entity_type,
                entity_id,
                detected_at,
                severity,
                deviation_pct,
                expected_value,
                actual_value,
                sample_size,
                enstr_code,
                kato_code,
                metadata,
                updated_at
            ) VALUES
            """,
            insert_rows,
        )

    return {
        "analyzed_categories": analyzed_categories,
        "anomalies_found": anomalies_found,
        "skipped_categories": skipped_categories,
    }


def main(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    force: bool = False,
) -> dict[str, int]:
    config = _load_config(config_path)
    clickhouse = _get_clickhouse_client()
    try:
        result = detect_price_anomalies(clickhouse, config, force=force)
        LOGGER.info(
            "price_anomaly_detection_result analyzed_categories=%d anomalies_found=%d skipped_categories=%d",
            result["analyzed_categories"],
            result["anomalies_found"],
            result["skipped_categories"],
        )
        return result
    finally:
        clickhouse.disconnect()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Price anomaly detection for contracts")
    _config_action = parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    _force_action = parser.add_argument(
        "--force",
        action="store_true",
        help="Truncate anomaly_results before detection",
    )
    _parser_actions = (_config_action, _force_action)
    _ = _parser_actions

    args = parser.parse_args()
    summary = main(config_path=cast(Path, args.config), force=cast(bool, args.force))
    LOGGER.info("Price anomaly summary: %s", summary)
