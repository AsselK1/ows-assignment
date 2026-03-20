from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Protocol, cast

import yaml  # type: ignore[import-untyped]  # pyright: ignore[reportMissingModuleSource]

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.api import BaseOWSClient, OWSAPIError, PLANS_ALL
from src.api.models import Plan

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_CHECKPOINT_PATH = Path(".sisyphus/state/etl_plans_checkpoint.json")
CHECKPOINT_REPLAY_WINDOW_SECONDS = 300
PLANS_V3 = f"/v3{PLANS_ALL}"
TARGET_YEARS = {2024, 2025, 2026}

Record = Mapping[str, object]
JSONScalar = str | int | float | bool | None
ParsedPlanRow = tuple[
    int,
    str,
    int,
    int,
    str,
    str,
    str,
    Decimal,
    Decimal,
    int,
    int,
    datetime,
    datetime,
]


class ClickHouseClient(Protocol):
    def query(self, query: str, parameters: dict | None = None) -> Any: ...

    def command(self, query: str, parameters: dict | list | None = None) -> None: ...

    def insert(
        self,
        table: str,
        data: Sequence[Sequence[object]],
        column_names: Sequence[str] | None = None,
    ) -> None: ...


@dataclass
class DataQualityStats:
    raw_records: int = 0
    parsed_records: int = 0
    inserted_records: int = 0
    unchanged_records: int = 0
    skipped_missing_id: int = 0
    skipped_non_target_bin: int = 0
    skipped_non_target_year: int = 0
    skipped_missing_enstr: int = 0
    duplicate_rows_in_batches: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "raw_records": self.raw_records,
            "parsed_records": self.parsed_records,
            "inserted_records": self.inserted_records,
            "unchanged_records": self.unchanged_records,
            "skipped_missing_id": self.skipped_missing_id,
            "skipped_non_target_bin": self.skipped_non_target_bin,
            "skipped_non_target_year": self.skipped_non_target_year,
            "skipped_missing_enstr": self.skipped_missing_enstr,
            "duplicate_rows_in_batches": self.duplicate_rows_in_batches,
        }


@dataclass
class CheckpointState:
    last_successful_sync: datetime | None
    max_seen_modified: datetime | None
    endpoint_index: int
    bin_index: int
    processed_records: int
    inserted_records: int


def _now_utc_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _datetime_to_iso(value: datetime) -> str:
    return value.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _iso_to_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)


def _parse_datetime(value: object, *, default: datetime) -> datetime:
    if value is None:
        return default

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=UTC).replace(tzinfo=None)

    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value).decode("utf-8", errors="ignore").strip()
    else:
        raw = str(value).strip()

    if not raw:
        return default

    try:
        return _iso_to_datetime(raw)
    except ValueError:
        pass

    patterns = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    )
    for pattern in patterns:
        try:
            parsed = datetime.strptime(raw, pattern)
            return parsed
        except ValueError:
            continue

    return default


def _get_clickhouse_client() -> ClickHouseClient:
    host = os.getenv("CLICKHOUSE_HOST")
    if not host:
        raise ValueError("CLICKHOUSE_HOST is required")

    port_raw = os.getenv("CLICKHOUSE_PORT", "8443")
    try:
        port = int(port_raw)
    except ValueError as error:
        raise ValueError(f"CLICKHOUSE_PORT must be an integer, got: {port_raw}") from error

    database = os.getenv("CLICKHOUSE_DB")
    if not database:
        raise ValueError("CLICKHOUSE_DB is required")

    user = os.getenv("CLICKHOUSE_USER", "default")
    password = os.getenv("CLICKHOUSE_PASSWORD", "")

    clickhouse_connect = importlib.import_module("clickhouse_connect")
    client_factory = cast(
        Callable[..., ClickHouseClient], getattr(clickhouse_connect, "get_client")
    )
    return client_factory(
        host=host,
        port=port,
        database=database,
        username=user,
        password=password,
        secure=True,
        connect_timeout=15,
        send_receive_timeout=30,
    )


def _to_uint64(value: object, default: int = 0) -> int:
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


def _to_uint16(value: object, default: int = 0) -> int:
    parsed = _to_uint64(value, default=default)
    return min(65535, parsed)


def _to_decimal(value: object, *, scale: int, default: Decimal | None = None) -> Decimal:
    fallback = default if default is not None else Decimal("0")
    if value is None:
        return fallback

    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, (int, str)):
        try:
            text = str(value).strip().replace(" ", "")
            if not text:
                return fallback
            if text.count(",") == 1 and text.count(".") == 0:
                text = text.replace(",", ".")
            parsed = Decimal(text)
        except (InvalidOperation, ValueError):
            return fallback
    elif isinstance(value, float):
        parsed = Decimal(str(value))
    else:
        return fallback

    quant = Decimal("1") if scale <= 0 else Decimal(f"1.{'0' * scale}")
    return parsed.quantize(quant, rounding=ROUND_HALF_UP)


def _first_non_empty_str(record: Record, keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _load_target_bins(config_path: Path) -> list[int]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    loaded_obj = cast(object, yaml.safe_load(config_path.read_text(encoding="utf-8")))
    if not isinstance(loaded_obj, dict):
        raise ValueError("config.yaml must be a dictionary at top level")
    loaded = cast(dict[str, object], loaded_obj)

    organizations = loaded.get("organizations")
    if not isinstance(organizations, dict):
        raise ValueError("config.yaml missing organizations section")
    organizations_dict = cast(dict[str, object], organizations)

    bins_raw = organizations_dict.get("bins")
    if not isinstance(bins_raw, list):
        raise ValueError("config.yaml organizations.bins must be a list")

    bins: list[int] = []
    bins_raw_items = cast(list[object], bins_raw)
    for item in bins_raw_items:
        parsed = _to_uint64(item, default=-1)
        if parsed < 0:
            continue
        bins.append(parsed)

    if len(bins) != 27:
        LOGGER.warning("Expected 27 BINs in config, got %d", len(bins))

    unique_bins = sorted(set(bins))
    if not unique_bins:
        raise ValueError("No valid BINs found in config.yaml")
    return unique_bins


def _read_checkpoint(path: Path) -> CheckpointState:
    if not path.exists():
        return CheckpointState(
            last_successful_sync=None,
            max_seen_modified=None,
            endpoint_index=0,
            bin_index=0,
            processed_records=0,
            inserted_records=0,
        )

    try:
        payload_obj = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Invalid checkpoint file %s: %s", path, error)
        return CheckpointState(None, None, 0, 0, 0, 0)

    if not isinstance(payload_obj, dict):
        return CheckpointState(None, None, 0, 0, 0, 0)
    payload = cast(dict[str, object], payload_obj)

    def _parse_iso_from_payload(key: str) -> datetime | None:
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            try:
                return _iso_to_datetime(raw)
            except ValueError:
                LOGGER.warning("Invalid %s in checkpoint: %s", key, raw)
        return None

    return CheckpointState(
        last_successful_sync=_parse_iso_from_payload("last_successful_sync"),
        max_seen_modified=_parse_iso_from_payload("max_seen_modified"),
        endpoint_index=_to_uint64(payload.get("endpoint_index"), default=0),
        bin_index=_to_uint64(payload.get("bin_index"), default=0),
        processed_records=_to_uint64(payload.get("processed_records"), default=0),
        inserted_records=_to_uint64(payload.get("inserted_records"), default=0),
    )


def _write_checkpoint(path: Path, state: CheckpointState, *, last_error: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "version": 1,
        "last_successful_sync": (
            _datetime_to_iso(state.last_successful_sync) if state.last_successful_sync else None
        ),
        "max_seen_modified": (
            _datetime_to_iso(state.max_seen_modified) if state.max_seen_modified else None
        ),
        "endpoint_index": state.endpoint_index,
        "bin_index": state.bin_index,
        "processed_records": state.processed_records,
        "inserted_records": state.inserted_records,
        "updated_at": _datetime_to_iso(_now_utc_naive()),
    }
    if last_error:
        payload["last_error"] = last_error
    written_chars = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.debug("Plans checkpoint saved to %s (%d chars)", path, written_chars)


def _compare_columns() -> tuple[str, ...]:
    return (
        "plan_number",
        "customer_bin",
        "organizer_bin",
        "enstr_code",
        "kato_code",
        "mkei_code",
        "planned_amount",
        "quantity",
        "ref_plan_status_id",
        "plan_year",
        "publish_date",
    )


def _fetch_existing_latest_by_ids(
    clickhouse: ClickHouseClient,
    ids: Sequence[int],
) -> dict[int, tuple[object, ...]]:
    if not ids:
        return {}

    compare_columns = ", ".join(_compare_columns())
    argmax_columns = ", ".join(
        [f"argMax({column}, updated_at) AS {column}" for column in _compare_columns()]
    )
    query = (
        "SELECT id, "
        f"{compare_columns} "
        "FROM ("
        "SELECT id, "
        f"{argmax_columns} "
        "FROM plans WHERE id IN %(ids)s GROUP BY id"
        ")"
    )
    result = clickhouse.query(query, {"ids": tuple(ids)})
    rows = result.result_rows
    existing: dict[int, tuple[object, ...]] = {}
    for row in rows:
        if len(row) < 2:
            continue
        row_id = _to_uint64(row[0], default=0)
        if row_id <= 0:
            continue
        existing[row_id] = tuple(row[1:])
    return existing


def _upsert_batch(
    clickhouse: ClickHouseClient, rows: Sequence[ParsedPlanRow]
) -> tuple[int, int, int]:
    if not rows:
        return 0, 0, 0

    latest_by_id: dict[int, ParsedPlanRow] = {}
    for row in rows:
        row_id = row[0]
        current = latest_by_id.get(row_id)
        if current is None or row[12] >= current[12]:
            latest_by_id[row_id] = row

    duplicate_count = len(rows) - len(latest_by_id)
    deduped_rows = list(latest_by_id.values())
    existing = _fetch_existing_latest_by_ids(clickhouse, [row[0] for row in deduped_rows])

    insert_rows: list[ParsedPlanRow] = []
    unchanged = 0
    for row in deduped_rows:
        row_id = row[0]
        comparable = tuple(row[1:12])
        if row_id in existing and existing[row_id] == comparable:
            unchanged += 1
            continue
        insert_rows.append(row)

    if not insert_rows:
        return 0, duplicate_count, unchanged

    columns = (
        "id",
        "plan_number",
        "customer_bin",
        "organizer_bin",
        "enstr_code",
        "kato_code",
        "mkei_code",
        "planned_amount",
        "quantity",
        "ref_plan_status_id",
        "plan_year",
        "publish_date",
        "updated_at",
    )
    clickhouse.insert(
        "plans",
        insert_rows,
        column_names=columns,
    )
    return len(insert_rows), duplicate_count, unchanged


def _incremental_params_candidates(since: datetime | None) -> list[dict[str, JSONScalar]]:
    if since is None:
        return [{}]

    marker = _datetime_to_iso(since)
    return [
        {"last_modified": marker},
        {"lastModified": marker},
        {"crdate": marker},
        {"crdate_from": marker},
        {"from_date": marker},
    ]


def _iterate_bin_endpoint(
    ows_client: BaseOWSClient,
    endpoint: str,
    bin_value: int,
    since: datetime | None,
) -> Iterator[Record]:
    base_params: dict[str, JSONScalar] = {"customer_bin": str(bin_value)}
    for candidate in _incremental_params_candidates(since):
        params = dict(base_params)
        params.update(candidate)
        try:
            LOGGER.debug("Requesting %s with params=%s", endpoint, params)
            for item in ows_client.paginate(endpoint=endpoint, params=params):
                yield cast(Record, item)
            return
        except OWSAPIError as error:
            if since is None:
                raise
            LOGGER.warning(
                "Incremental params not accepted for %s and BIN %s (%s): %s",
                endpoint,
                bin_value,
                params,
                error,
            )

    if since is not None:
        LOGGER.info(
            "Falling back to full-per-BIN fetch for endpoint %s and BIN %s",
            endpoint,
            bin_value,
        )
    for item in ows_client.paginate(endpoint=endpoint, params=base_params):
        yield cast(Record, item)


def _stable_uint64(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _extract_point_list(record: Record) -> list[Record]:
    points_raw = record.get("point_list", record.get("pointList", record.get("points")))
    if not isinstance(points_raw, list):
        return [record]

    points: list[Record] = []
    points_raw_items = cast(list[object], points_raw)
    for point in points_raw_items:
        if isinstance(point, Mapping):
            points.append(cast(Record, point))

    return points if points else [record]


def _derive_row_id(plan_id: int, point: Record, point_index: int, has_point_list: bool) -> int:
    if not has_point_list and point_index == 0:
        return plan_id

    point_id = _to_uint64(
        point.get("id", point.get("point_id", point.get("pointId", point.get("plan_point_id")))),
        default=0,
    )
    if point_id > 0:
        return _stable_uint64(f"{plan_id}:{point_id}")

    signature = "|".join(
        [
            str(plan_id),
            str(point_index),
            _first_non_empty_str(
                point, ("ref_enstru_code", "enstr_code", "enstru_code", "tru_code", "code")
            ),
            _first_non_empty_str(point, ("kato_code", "kato", "katoCode")),
            _first_non_empty_str(point, ("mkei_code", "mkei", "unit_code", "measure_code")),
            str(point.get("planned_amount", point.get("amount", point.get("sum")))),
            str(point.get("quantity", point.get("qty", point.get("count")))),
        ]
    )
    return _stable_uint64(signature)


def _parse_plan_record(
    record: Record,
    target_bins: set[int],
    stats: DataQualityStats,
    fallback_customer_bin: int,
) -> tuple[list[ParsedPlanRow], datetime | None]:
    stats.raw_records += 1

    plan_id = _to_uint64(record.get("id"), default=0)
    if plan_id <= 0:
        stats.skipped_missing_id += 1
        return [], None

    customer_bin = _to_uint64(
        record.get("customer_bin", record.get("customerBin", record.get("bin"))),
        default=fallback_customer_bin,
    )
    if customer_bin <= 0 or customer_bin not in target_bins:
        stats.skipped_non_target_bin += 1
        return [], None

    now = _now_utc_naive()
    publish_date = _parse_datetime(
        record.get("publish_date", record.get("publishDate", record.get("crdate"))),
        default=now,
    )
    updated_at = _parse_datetime(
        record.get(
            "last_modified",
            record.get("lastModified", record.get("updated_at", record.get("crdate"))),
        ),
        default=publish_date,
    )

    plan_year = _to_uint16(
        record.get("plan_year", record.get("year", record.get("financial_year"))),
        default=publish_date.year,
    )
    if plan_year not in TARGET_YEARS:
        stats.skipped_non_target_year += 1
        return [], updated_at

    plan_number = _first_non_empty_str(
        record,
        ("plan_number", "planNumber", "number", "number_anno", "num"),
    )
    organizer_bin = _to_uint64(record.get("organizer_bin", record.get("organizerBin")), default=0)
    ref_plan_status_id = _to_uint64(
        record.get("ref_plan_status_id", record.get("status_id", record.get("statusId"))),
        default=0,
    )

    raw_point_list = record.get("point_list", record.get("pointList", record.get("points")))
    has_point_list = isinstance(raw_point_list, list)
    points = _extract_point_list(record)
    rows: list[ParsedPlanRow] = []

    for point_index, point in enumerate(points):
        point_enstr = _first_non_empty_str(
            point,
            ("ref_enstru_code", "enstr_code", "enstru_code", "tru_code", "code", "spec_enstr_code"),
        )
        enstr_code = point_enstr or _first_non_empty_str(
            record,
            ("ref_enstru_code", "enstr_code", "enstru_code", "tru_code", "code", "spec_enstr_code"),
        )
        if not enstr_code:
            stats.skipped_missing_enstr += 1
            continue

        kato_code = _first_non_empty_str(
            point, ("kato_code", "kato", "katoCode")
        ) or _first_non_empty_str(
            record,
            ("kato_code", "kato", "katoCode"),
        )
        mkei_code = _first_non_empty_str(
            point,
            ("mkei_code", "mkei", "unit_code", "unitCode", "measure_code"),
        ) or _first_non_empty_str(
            record,
            ("mkei_code", "mkei", "unit_code", "unitCode", "measure_code"),
        )

        planned_amount = _to_decimal(
            point.get(
                "planned_amount",
                point.get(
                    "amount", point.get("sum", record.get("planned_amount", record.get("amount")))
                ),
            ),
            scale=2,
            default=Decimal("0.00"),
        )
        quantity = _to_decimal(
            point.get("quantity", point.get("qty", point.get("count", record.get("quantity")))),
            scale=3,
            default=Decimal("0.000"),
        )

        parsed_plan = Plan.model_validate(
            {
                "id": plan_id,
                "customer_bin": str(customer_bin),
                "year": plan_year,
                "enstr_code": enstr_code,
            }
        )

        row_id = _derive_row_id(plan_id, point, point_index, has_point_list)
        row: ParsedPlanRow = (
            row_id,
            plan_number,
            _to_uint64(parsed_plan.customer_bin, default=customer_bin),
            organizer_bin,
            parsed_plan.enstr_code or "",
            kato_code,
            mkei_code,
            planned_amount,
            quantity,
            ref_plan_status_id,
            _to_uint16(parsed_plan.year, default=plan_year),
            publish_date,
            updated_at,
        )
        rows.append(row)
        stats.parsed_records += 1

    return rows, updated_at


def load_plans(
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient,
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> dict[str, int]:
    target_bins = set(_load_target_bins(config_path))
    checkpoint = _read_checkpoint(checkpoint_path)

    since: datetime | None = None
    if not force:
        source = checkpoint.max_seen_modified or checkpoint.last_successful_sync
        if source is not None:
            since = source - timedelta(seconds=CHECKPOINT_REPLAY_WINDOW_SECONDS)

    endpoints = (PLANS_V3,)
    bins_ordered = sorted(target_bins)
    stats = DataQualityStats()

    if force:
        LOGGER.info("Force mode enabled: loading all plan records for configured BINs")
        checkpoint.endpoint_index = 0
        checkpoint.bin_index = 0

    start_endpoint_index = checkpoint.endpoint_index if since is not None and not force else 0
    start_bin_index = checkpoint.bin_index if since is not None and not force else 0

    batch_rows: list[ParsedPlanRow] = []
    max_seen = checkpoint.max_seen_modified
    if max_seen is None:
        max_seen = checkpoint.last_successful_sync

    try:
        for endpoint_index, endpoint in enumerate(endpoints):
            if endpoint_index < start_endpoint_index:
                continue

            current_start_bin = start_bin_index if endpoint_index == start_endpoint_index else 0
            for bin_index, bin_value in enumerate(bins_ordered):
                if bin_index < current_start_bin:
                    continue

                checkpoint.endpoint_index = endpoint_index
                checkpoint.bin_index = bin_index
                _write_checkpoint(checkpoint_path, checkpoint)

                for item in _iterate_bin_endpoint(ows_client, endpoint, bin_value, since):
                    rows, modified_at = _parse_plan_record(item, target_bins, stats, bin_value)
                    if modified_at is not None and (max_seen is None or modified_at > max_seen):
                        max_seen = modified_at

                    if not rows:
                        continue

                    batch_rows.extend(rows)
                    while len(batch_rows) >= batch_size:
                        current_batch = batch_rows[:batch_size]
                        inserted, duplicates, unchanged = _upsert_batch(clickhouse, current_batch)
                        stats.inserted_records += inserted
                        stats.duplicate_rows_in_batches += duplicates
                        stats.unchanged_records += unchanged
                        checkpoint.processed_records = stats.parsed_records
                        checkpoint.inserted_records = stats.inserted_records
                        checkpoint.max_seen_modified = max_seen
                        _write_checkpoint(checkpoint_path, checkpoint)
                        LOGGER.info(
                            "Plans progress: parsed=%d inserted=%d duplicates=%d unchanged=%d",
                            stats.parsed_records,
                            stats.inserted_records,
                            stats.duplicate_rows_in_batches,
                            stats.unchanged_records,
                        )
                        batch_rows = batch_rows[batch_size:]

        if batch_rows:
            inserted, duplicates, unchanged = _upsert_batch(clickhouse, batch_rows)
            stats.inserted_records += inserted
            stats.duplicate_rows_in_batches += duplicates
            stats.unchanged_records += unchanged
            checkpoint.processed_records = stats.parsed_records
            checkpoint.inserted_records = stats.inserted_records
            checkpoint.max_seen_modified = max_seen
            _write_checkpoint(checkpoint_path, checkpoint)

        checkpoint.last_successful_sync = max_seen or _now_utc_naive()
        checkpoint.max_seen_modified = checkpoint.last_successful_sync
        checkpoint.endpoint_index = 0
        checkpoint.bin_index = 0
        _write_checkpoint(checkpoint_path, checkpoint)
    except Exception as error:  # noqa: BLE001
        checkpoint.max_seen_modified = max_seen
        checkpoint.processed_records = stats.parsed_records
        checkpoint.inserted_records = stats.inserted_records
        _write_checkpoint(checkpoint_path, checkpoint, last_error=str(error))
        LOGGER.error("Plans ETL failed; checkpoint persisted for recovery: %s", error)
        raise

    quality = stats.as_dict()
    LOGGER.info("Plans ETL finished with quality stats: %s", quality)
    return quality


def main(
    *,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    config_path: Path = DEFAULT_CONFIG_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> dict[str, int]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    ows_client = BaseOWSClient()
    clickhouse = _get_clickhouse_client()
    return load_plans(
        ows_client,
        clickhouse,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        force=force,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for OWS plans endpoint")
    force_action = parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore incremental marker and run full per-BIN fetch",
    )
    batch_action = parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Insert batch size (default: 500)",
    )
    config_action = parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml with organizations.bins",
    )
    checkpoint_action = parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to checkpoint JSON file",
    )
    _cli_actions = (force_action, batch_action, config_action, checkpoint_action)
    LOGGER.debug("Registered %d CLI arguments for plans ETL", len(_cli_actions))

    args = parser.parse_args()
    result: dict[str, int] = main(
        force=cast(bool, args.force),
        batch_size=cast(int, args.batch_size),
        config_path=cast(Path, args.config),
        checkpoint_path=cast(Path, args.checkpoint),
    )
    LOGGER.debug("Plans ETL CLI result keys: %s", sorted(result.keys()))
