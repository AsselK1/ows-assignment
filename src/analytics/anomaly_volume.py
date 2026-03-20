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
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Protocol, cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_MIN_SAMPLE_SIZE = 5
BASELINE_START_YEAR = 2024
BASELINE_END_YEAR = 2026
ANOMALY_MULTIPLIER_THRESHOLD = Decimal("3.000")

SMALL_ORG_THRESHOLD = Decimal("1000000.000")
MEDIUM_ORG_THRESHOLD = Decimal("2000000.000")
LARGE_ORG_THRESHOLD = Decimal("5000000.000")


class ClickHouseClient(Protocol):
    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]: ...

    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class ContractQuantitySnapshot:
    contract_id: int
    entity_type: str
    entity_id: int
    customer_bin: int
    enstr_code_lvl4: str
    kato_code: str | None
    sign_date: datetime
    year: int
    month: int
    quantity: Decimal


@dataclass(frozen=True)
class OrgSizeProfile:
    customer_bin: int
    label: str
    threshold: Decimal


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
    scale: int = 3,
    default: Decimal | None = None,
) -> Decimal:
    fallback = default if default is not None else Decimal("0.000")
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


def _normalize_enstr_lvl4(value: object) -> str:
    raw = str(value if value is not None else "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits[:4]


def _normalize_kato(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    return digits


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


def _load_contract_quantity_rows(clickhouse_client: ClickHouseClient) -> list[tuple[object, ...]]:
    return clickhouse_client.execute(
        """
        SELECT
            c.id,
            c.customer_bin,
            c.lot_id,
            c.enstr_code_lvl4,
            c.kato_code,
            c.sign_date,
            l.quantity
        FROM
        (
            SELECT
                id,
                argMax(customer_bin, updated_at) AS customer_bin,
                argMax(lot_id, updated_at) AS lot_id,
                argMax(enstr_code_lvl4, updated_at) AS enstr_code_lvl4,
                argMax(kato_code, updated_at) AS kato_code,
                argMax(sign_date, updated_at) AS sign_date,
                argMax(contract_sum, updated_at) AS contract_sum
            FROM contracts
            GROUP BY id
        ) AS c
        LEFT JOIN
        (
            SELECT
                id,
                argMax(quantity, updated_at) AS quantity
            FROM lots
            GROUP BY id
        ) AS l
        ON c.lot_id = l.id
        WHERE c.contract_sum > 0
          AND toYear(c.sign_date) BETWEEN %(start_year)s AND %(end_year)s
        """,
        {"start_year": BASELINE_START_YEAR, "end_year": BASELINE_END_YEAR},
    )


def _load_org_name_by_bin(clickhouse_client: ClickHouseClient) -> dict[int, str]:
    rows = clickhouse_client.execute(
        """
        SELECT
            bin,
            argMax(name_ru, updated_at) AS name_ru,
            argMax(name_kz, updated_at) AS name_kz
        FROM subjects
        GROUP BY bin
        """
    )
    result: dict[int, str] = {}
    for row in rows:
        if len(row) != 3:
            continue
        customer_bin = _to_uint64(row[0], default=0)
        if customer_bin <= 0:
            continue
        name_ru = str(row[1] if row[1] is not None else "").strip()
        name_kz = str(row[2] if row[2] is not None else "").strip()
        combined = " ".join(part for part in (name_ru, name_kz) if part)
        result[customer_bin] = combined
    return result


def _load_customer_volume_stats(
    clickhouse_client: ClickHouseClient,
) -> dict[int, tuple[int, Decimal]]:
    rows = clickhouse_client.execute(
        """
        SELECT
            customer_bin,
            count() AS contracts_count,
            sum(contract_sum) AS total_contract_sum
        FROM
        (
            SELECT
                id,
                argMax(customer_bin, updated_at) AS customer_bin,
                argMax(contract_sum, updated_at) AS contract_sum,
                argMax(sign_date, updated_at) AS sign_date
            FROM contracts
            GROUP BY id
        )
        WHERE toYear(sign_date) BETWEEN %(start_year)s AND %(end_year)s
          AND contract_sum > 0
        GROUP BY customer_bin
        """,
        {"start_year": BASELINE_START_YEAR, "end_year": BASELINE_END_YEAR},
    )
    stats: dict[int, tuple[int, Decimal]] = {}
    for row in rows:
        if len(row) != 3:
            continue
        customer_bin = _to_uint64(row[0], default=0)
        if customer_bin <= 0:
            continue
        contracts_count = _to_uint64(row[1], default=0)
        total_contract_sum = _to_decimal(row[2], scale=2, default=Decimal("0.00"))
        stats[customer_bin] = (contracts_count, total_contract_sum)
    return stats


def _contains_any_token(text: str, tokens: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in tokens)


def _resolve_org_thresholds(config: Mapping[str, object]) -> dict[str, Decimal]:
    analytics_cfg = _as_object_mapping(config.get("analytics")) or {}
    thresholds_cfg = _as_object_mapping(analytics_cfg.get("volume_org_thresholds")) or {}
    return {
        "small": _to_decimal(
            thresholds_cfg.get("small"),
            scale=3,
            default=SMALL_ORG_THRESHOLD,
        ),
        "medium": _to_decimal(
            thresholds_cfg.get("medium"),
            scale=3,
            default=MEDIUM_ORG_THRESHOLD,
        ),
        "large": _to_decimal(
            thresholds_cfg.get("large"),
            scale=3,
            default=LARGE_ORG_THRESHOLD,
        ),
    }


def _build_org_profiles(
    contract_snapshots: Sequence[ContractQuantitySnapshot],
    org_names: Mapping[int, str],
    customer_stats: Mapping[int, tuple[int, Decimal]],
    thresholds: Mapping[str, Decimal],
) -> dict[int, OrgSizeProfile]:
    customer_bins = {
        snapshot.customer_bin for snapshot in contract_snapshots if snapshot.customer_bin > 0
    }
    profiles: dict[int, OrgSizeProfile] = {}

    small_tokens = (
        "школ",
        "school",
        "лицей",
        "гимназ",
        "детск",
        "kindergarten",
        "ясли",
        "библиот",
        "museum",
        "амбулат",
        "clinic",
        "поликлиник",
    )
    large_tokens = (
        "министер",
        "ministry",
        "акимат",
        "управлен",
        "департамент",
        "комитет",
        "national",
        "националь",
    )

    for customer_bin in customer_bins:
        contracts_count, total_contract_sum = customer_stats.get(
            customer_bin,
            (0, Decimal("0.00")),
        )
        org_name = org_names.get(customer_bin, "")

        label = "small"
        if contracts_count >= 300 or total_contract_sum >= Decimal("5000000000.00"):
            label = "large"
        elif contracts_count >= 80 or total_contract_sum >= Decimal("500000000.00"):
            label = "medium"

        if _contains_any_token(org_name, large_tokens):
            if label == "small":
                label = "medium"
        if _contains_any_token(org_name, small_tokens):
            if label != "large":
                label = "small"

        threshold = thresholds.get(label)
        if threshold is None:
            threshold = SMALL_ORG_THRESHOLD

        profiles[customer_bin] = OrgSizeProfile(
            customer_bin=customer_bin,
            label=label,
            threshold=threshold,
        )

    return profiles


def _average_decimal(values: Sequence[Decimal], *, scale: int = 3) -> Decimal:
    if not values:
        return Decimal("0.000")
    total = sum(values, Decimal("0"))
    avg = total / Decimal(len(values))
    quant = Decimal("1") if scale <= 0 else Decimal(f"1.{'0' * scale}")
    return avg.quantize(quant, rounding=ROUND_HALF_UP)


def _classify_severity(multiplier_like: Decimal) -> str:
    if multiplier_like > Decimal("10.000"):
        return "high"
    if multiplier_like >= Decimal("5.000"):
        return "medium"
    return "low"


def _build_snapshot(row: tuple[object, ...], now_dt: datetime) -> ContractQuantitySnapshot | None:
    if len(row) != 7:
        return None
    contract_id = _to_uint64(row[0], default=0)
    if contract_id <= 0:
        return None

    customer_bin = _to_uint64(row[1], default=0)
    lot_id = _to_uint64(row[2], default=0)
    enstr_code_lvl4 = _normalize_enstr_lvl4(row[3])
    if len(enstr_code_lvl4) < 4:
        return None

    sign_date = _parse_datetime(row[5], default=now_dt)
    year = sign_date.year
    if year < BASELINE_START_YEAR or year > BASELINE_END_YEAR:
        return None
    month = sign_date.month

    quantity = _to_decimal(row[6], scale=3, default=Decimal("0.000"))
    if quantity <= 0:
        return None

    entity_type = "lot" if lot_id > 0 else "contract"
    entity_id = lot_id if lot_id > 0 else contract_id
    if entity_id <= 0:
        return None

    return ContractQuantitySnapshot(
        contract_id=contract_id,
        entity_type=entity_type,
        entity_id=entity_id,
        customer_bin=customer_bin,
        enstr_code_lvl4=enstr_code_lvl4,
        kato_code=_normalize_kato(row[4]),
        sign_date=sign_date,
        year=year,
        month=month,
        quantity=quantity,
    )


def detect_volume_anomalies(
    clickhouse_client: ClickHouseClient,
    config: Mapping[str, object],
    *,
    force: bool = False,
) -> dict[str, int]:
    _ensure_anomaly_table(clickhouse_client)
    if force:
        _ = clickhouse_client.execute(
            "ALTER TABLE anomaly_results DELETE WHERE anomaly_type = 'volume'"
        )

    analytics_cfg = _as_object_mapping(config.get("analytics")) or {}
    min_sample_size_raw = _to_uint64(
        analytics_cfg.get("min_sample_size"), default=DEFAULT_MIN_SAMPLE_SIZE
    )
    min_sample_size = max(DEFAULT_MIN_SAMPLE_SIZE, min_sample_size_raw)
    org_thresholds = _resolve_org_thresholds(config)

    rows = _load_contract_quantity_rows(clickhouse_client)
    now_dt = datetime.now(UTC).replace(tzinfo=None)
    snapshots: list[ContractQuantitySnapshot] = []
    for row in rows:
        snapshot = _build_snapshot(row, now_dt)
        if snapshot is None:
            continue
        snapshots.append(snapshot)

    org_names = _load_org_name_by_bin(clickhouse_client)
    customer_stats = _load_customer_volume_stats(clickhouse_client)
    org_profiles = _build_org_profiles(snapshots, org_names, customer_stats, org_thresholds)

    by_enstr: dict[str, list[ContractQuantitySnapshot]] = defaultdict(list)
    by_enstr_month: dict[tuple[str, int], list[ContractQuantitySnapshot]] = defaultdict(list)
    for snapshot in snapshots:
        by_enstr[snapshot.enstr_code_lvl4].append(snapshot)
        by_enstr_month[(snapshot.enstr_code_lvl4, snapshot.month)].append(snapshot)

    analyzed_contracts = 0
    skipped_contracts = 0
    anomalies_found = 0
    detected_at = now_dt

    insert_rows: list[tuple[object, ...]] = []
    warned_small_samples: set[tuple[str, int, int]] = set()

    for snapshot in snapshots:
        seasonal_key = (snapshot.enstr_code_lvl4, snapshot.month)
        seasonal_candidates = [
            candidate.quantity
            for candidate in by_enstr_month.get(seasonal_key, [])
            if candidate.year < snapshot.year
        ]
        seasonal_sample_size = len(seasonal_candidates)

        annual_candidates = [
            candidate.quantity
            for candidate in by_enstr.get(snapshot.enstr_code_lvl4, [])
            if candidate.year < snapshot.year
        ]
        annual_sample_size = len(annual_candidates)

        seasonal_avg = _average_decimal(seasonal_candidates, scale=3)
        annual_avg = _average_decimal(annual_candidates, scale=3)

        baseline_mode = "seasonal"
        baseline_sample_size = seasonal_sample_size
        historical_avg = seasonal_avg
        if seasonal_sample_size < min_sample_size:
            sample_warning_key = (snapshot.enstr_code_lvl4, snapshot.month, snapshot.year)
            if sample_warning_key not in warned_small_samples:
                LOGGER.warning(
                    (
                        "Seasonal sample below minimum; using annual fallback: "
                        "enstr=%s month=%d year=%d seasonal_n=%d min_n=%d"
                    ),
                    snapshot.enstr_code_lvl4,
                    snapshot.month,
                    snapshot.year,
                    seasonal_sample_size,
                    min_sample_size,
                )
                warned_small_samples.add(sample_warning_key)

            baseline_mode = "annual_fallback"
            baseline_sample_size = annual_sample_size
            historical_avg = annual_avg

        if baseline_sample_size < min_sample_size or historical_avg <= 0:
            skipped_contracts += 1
            continue

        analyzed_contracts += 1
        multiplier = (snapshot.quantity / historical_avg).quantize(
            Decimal("1.000"),
            rounding=ROUND_HALF_UP,
        )
        deviation_pct = ((multiplier - Decimal("1.000")) * Decimal("100")).quantize(
            Decimal("1.00"),
            rounding=ROUND_HALF_UP,
        )

        profile = org_profiles.get(snapshot.customer_bin)
        if profile is None:
            profile = OrgSizeProfile(
                customer_bin=snapshot.customer_bin,
                label="small",
                threshold=org_thresholds["small"],
            )

        org_ratio = (snapshot.quantity / profile.threshold).quantize(
            Decimal("1.000"),
            rounding=ROUND_HALF_UP,
        )
        org_breach = snapshot.quantity > profile.threshold
        org_size_check = "passed"
        if org_breach:
            org_size_check = f"failed:{profile.label}_threshold_exceeded"

        multiplier_trigger = multiplier > ANOMALY_MULTIPLIER_THRESHOLD
        if not multiplier_trigger and not org_breach:
            continue

        severity_driver = multiplier if multiplier > org_ratio else org_ratio
        severity = _classify_severity(severity_driver)
        anomaly_id = _stable_uint64(
            [
                "volume",
                snapshot.entity_type,
                snapshot.entity_id,
                snapshot.enstr_code_lvl4,
                snapshot.sign_date.isoformat(),
                detected_at.isoformat(),
            ]
        )

        metadata = {
            "actual_quantity": str(snapshot.quantity.quantize(Decimal("1.000"))),
            "annual_avg": str(annual_avg),
            "baseline_mode": baseline_mode,
            "baseline_sample_size": baseline_sample_size,
            "historical_avg": str(historical_avg),
            "month": snapshot.month,
            "multiplier": str(multiplier),
            "org_size_check": org_size_check,
            "org_size_label": profile.label,
            "org_size_threshold": str(profile.threshold.quantize(Decimal("1.000"))),
            "seasonal_avg": str(seasonal_avg),
            "seasonal_sample_size": seasonal_sample_size,
        }

        insert_rows.append(
            (
                anomaly_id,
                "volume",
                snapshot.entity_type,
                snapshot.entity_id,
                detected_at,
                severity,
                deviation_pct,
                historical_avg.quantize(Decimal("1.00"), rounding=ROUND_HALF_UP),
                snapshot.quantity.quantize(Decimal("1.00"), rounding=ROUND_HALF_UP),
                baseline_sample_size,
                snapshot.enstr_code_lvl4,
                snapshot.kato_code,
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
        "analyzed_contracts": analyzed_contracts,
        "anomalies_found": anomalies_found,
        "skipped_contracts": skipped_contracts,
    }


def main(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    force: bool = False,
) -> dict[str, int]:
    config = _load_config(config_path)
    clickhouse = _get_clickhouse_client()
    try:
        result = detect_volume_anomalies(clickhouse, config, force=force)
        LOGGER.info(
            "volume_detection_result analyzed_contracts=%d anomalies_found=%d skipped_contracts=%d",
            result["analyzed_contracts"],
            result["anomalies_found"],
            result["skipped_contracts"],
        )
        return result
    finally:
        clickhouse.disconnect()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Volume anomaly detection for contracts/lots")
    _config_action = parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    _force_action = parser.add_argument(
        "--force",
        action="store_true",
        help="Delete volume anomalies before detection",
    )
    _parser_actions = (_config_action, _force_action)
    _ = _parser_actions

    args = parser.parse_args()
    summary = main(config_path=cast(Path, args.config), force=cast(bool, args.force))
    LOGGER.info("Volume summary: %s", summary)
