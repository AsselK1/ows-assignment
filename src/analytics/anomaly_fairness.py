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

from src.analytics.statistics import median

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_MIN_SAMPLE_SIZE = 5
LOWER_FAIRNESS_BOUND = Decimal("0.7000")
UPPER_FAIRNESS_BOUND = Decimal("1.3000")


class ClickHouseClient(Protocol):
    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]: ...

    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class ContractSnapshot:
    contract_id: int
    contract_sum: Decimal
    sign_date: datetime
    quarter_start: datetime
    enstr_code_lvl4: str
    kato_code: str | None
    kato_region: str | None


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
            contract_sum,
            sign_date,
            toStartOfQuarter(sign_date) AS sign_quarter,
            enstr_code_lvl4,
            kato_code
        FROM
        (
            SELECT
                id,
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


def _kato_region_prefix(kato_code: str | None) -> str | None:
    if not kato_code:
        return None
    if len(kato_code) < 2:
        return None
    return kato_code[:2]


def _quarter_label(value: datetime) -> str:
    quarter = ((value.month - 1) // 3) + 1
    return f"{value.year}-Q{quarter}"


def _quarter_index(value: datetime) -> int:
    quarter = ((value.month - 1) // 3) + 1
    return (value.year * 4) + (quarter - 1)


def _quarter_distance(left: datetime, right: datetime) -> int:
    return abs(_quarter_index(left) - _quarter_index(right))


def _classify_fairness_severity(score: Decimal) -> str:
    if score < Decimal("0.5000") or score > Decimal("2.0000"):
        return "high"
    if score < Decimal("0.6000") or score > Decimal("1.5000"):
        return "medium"
    return "low"


def _build_contract_snapshot(row: tuple[object, ...], now_dt: datetime) -> ContractSnapshot | None:
    if len(row) != 6:
        return None
    contract_id = _to_uint64(row[0], default=0)
    if contract_id <= 0:
        return None
    contract_sum = _to_decimal(row[1], scale=2, default=Decimal("0.00"))
    if contract_sum <= 0:
        return None

    sign_date = _parse_datetime(row[2], default=now_dt)
    quarter_start = _parse_datetime(row[3], default=sign_date)
    enstr_code_lvl4 = _normalize_enstr_lvl4(row[4])
    if len(enstr_code_lvl4) < 4:
        return None

    kato_code = _normalize_kato(row[5])
    return ContractSnapshot(
        contract_id=contract_id,
        contract_sum=contract_sum,
        sign_date=sign_date,
        quarter_start=quarter_start,
        enstr_code_lvl4=enstr_code_lvl4,
        kato_code=kato_code,
        kato_region=_kato_region_prefix(kato_code),
    )


def _region_matches(
    candidates: Sequence[ContractSnapshot],
    target_region: str | None,
) -> list[ContractSnapshot]:
    if target_region is None:
        return []
    return [candidate for candidate in candidates if candidate.kato_region == target_region]


def _exact_kato_matches(
    candidates: Sequence[ContractSnapshot],
    target_kato: str | None,
) -> list[ContractSnapshot]:
    if target_kato is None:
        return []
    return [candidate for candidate in candidates if candidate.kato_code == target_kato]


def _select_regional_candidates(
    candidates: Sequence[ContractSnapshot],
    target: ContractSnapshot,
    *,
    min_sample_size: int,
) -> tuple[list[ContractSnapshot], str]:
    exact = _exact_kato_matches(candidates, target.kato_code)
    if len(exact) >= min_sample_size:
        return exact, "exact_kato"

    region = _region_matches(candidates, target.kato_region)
    if len(region) >= min_sample_size:
        return region, "regional_kato_prefix"

    all_enstr = list(candidates)
    return all_enstr, "enstr_only"


def _average_decimal(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal("0.00")
    total = sum(values, Decimal("0"))
    return total / Decimal(len(values))


def detect_fairness_anomalies(
    clickhouse_client: ClickHouseClient,
    config: Mapping[str, object],
    *,
    force: bool = False,
) -> dict[str, int]:
    _ensure_anomaly_table(clickhouse_client)
    if force:
        _ = clickhouse_client.execute(
            "ALTER TABLE anomaly_results DELETE WHERE anomaly_type = 'fairness'"
        )

    analytics_cfg = _as_object_mapping(config.get("analytics")) or {}
    min_sample_size_raw = _to_uint64(
        analytics_cfg.get("min_sample_size"), default=DEFAULT_MIN_SAMPLE_SIZE
    )
    min_sample_size = max(DEFAULT_MIN_SAMPLE_SIZE, min_sample_size_raw)

    rows = _load_latest_contracts(clickhouse_client)
    now_dt = datetime.now(UTC).replace(tzinfo=None)

    grouped_by_enstr: dict[str, list[ContractSnapshot]] = defaultdict(list)
    for row in rows:
        snapshot = _build_contract_snapshot(row, now_dt)
        if snapshot is None:
            continue
        grouped_by_enstr[snapshot.enstr_code_lvl4].append(snapshot)

    analyzed_contracts = 0
    skipped_contracts = 0
    anomalies_found = 0
    detected_at = now_dt
    insert_rows: list[tuple[object, ...]] = []

    for enstr_code, contracts in grouped_by_enstr.items():
        for target in contracts:
            same_enstr_others = [
                candidate for candidate in contracts if candidate.contract_id != target.contract_id
            ]
            same_quarter_pool = [
                candidate
                for candidate in same_enstr_others
                if _quarter_distance(candidate.quarter_start, target.quarter_start) == 0
            ]

            selected, regional_scope = _select_regional_candidates(
                same_quarter_pool,
                target,
                min_sample_size=min_sample_size,
            )
            temporal_scope = "same_quarter"

            if len(selected) < min_sample_size:
                temporal_pool = [
                    candidate
                    for candidate in same_enstr_others
                    if _quarter_distance(candidate.quarter_start, target.quarter_start) <= 1
                ]
                selected, regional_scope = _select_regional_candidates(
                    temporal_pool,
                    target,
                    min_sample_size=min_sample_size,
                )
                temporal_scope = "plus_minus_one_quarter"

            comparison_set_size = len(selected)
            if comparison_set_size < min_sample_size:
                skipped_contracts += 1
                LOGGER.warning(
                    (
                        "Skipping contract due to insufficient fairness comparison set: "
                        "contract_id=%d enstr=%s n=%d min_n=%d"
                    ),
                    target.contract_id,
                    enstr_code,
                    comparison_set_size,
                    min_sample_size,
                )
                continue

            analyzed_contracts += 1
            comparable_prices = [candidate.contract_sum for candidate in selected]
            median_price = median(comparable_prices)
            if median_price <= 0:
                skipped_contracts += 1
                LOGGER.warning(
                    "Skipping contract due to non-positive fairness median: contract_id=%d enstr=%s",
                    target.contract_id,
                    enstr_code,
                )
                continue

            avg_price = _average_decimal(comparable_prices)
            fairness_score = target.contract_sum / median_price

            if LOWER_FAIRNESS_BOUND <= fairness_score <= UPPER_FAIRNESS_BOUND:
                continue

            severity = _classify_fairness_severity(fairness_score)
            deviation_pct = (abs(fairness_score - Decimal("1")) * Decimal("100")).quantize(
                Decimal("1.00"),
                rounding=ROUND_HALF_UP,
            )
            median_price_q = median_price.quantize(Decimal("1.00"), rounding=ROUND_HALF_UP)
            avg_price_q = avg_price.quantize(Decimal("1.00"), rounding=ROUND_HALF_UP)
            fairness_score_q = fairness_score.quantize(Decimal("1.0000"), rounding=ROUND_HALF_UP)

            regional_matches = len(_region_matches(selected, target.kato_region))
            temporal_matches = sum(
                1
                for candidate in selected
                if (
                    _quarter_distance(candidate.quarter_start, target.quarter_start)
                    <= (0 if temporal_scope == "same_quarter" else 1)
                )
            )

            anomaly_id = _stable_uint64(
                ["fairness", "contract", target.contract_id, enstr_code, detected_at.isoformat()]
            )
            metadata = {
                "comparison_set_size": comparison_set_size,
                "median_price": str(median_price_q),
                "avg_price": str(avg_price_q),
                "fairness_score": str(fairness_score_q),
                "regional_matches": regional_matches,
                "temporal_matches": temporal_matches,
                "quarter": _quarter_label(target.quarter_start),
                "regional_scope": regional_scope,
                "temporal_scope": temporal_scope,
            }

            insert_rows.append(
                (
                    anomaly_id,
                    "fairness",
                    "contract",
                    target.contract_id,
                    detected_at,
                    severity,
                    deviation_pct,
                    median_price_q,
                    target.contract_sum.quantize(Decimal("1.00"), rounding=ROUND_HALF_UP),
                    comparison_set_size,
                    enstr_code,
                    target.kato_code,
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
        result = detect_fairness_anomalies(clickhouse, config, force=force)
        LOGGER.info(
            (
                "fairness_detection_result analyzed_contracts=%d "
                "anomalies_found=%d skipped_contracts=%d"
            ),
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

    parser = argparse.ArgumentParser(description="Fairness anomaly detection for contracts")
    _config_action = parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    _force_action = parser.add_argument(
        "--force",
        action="store_true",
        help="Delete fairness anomalies before detection",
    )
    _parser_actions = (_config_action, _force_action)
    _ = _parser_actions

    args = parser.parse_args()
    summary = main(config_path=cast(Path, args.config), force=cast(bool, args.force))
    LOGGER.info("Fairness summary: %s", summary)
