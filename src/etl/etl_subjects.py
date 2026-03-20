from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Protocol, cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.api import BaseOWSClient, OWSAPIError, SUBJECTS
from src.api.models import Subject

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_CHECKPOINT_PATH = Path(".sisyphus/state/etl_subjects_checkpoint.json")
CHECKPOINT_REPLAY_WINDOW_SECONDS = 300
SUBJECTS_ALL = f"{SUBJECTS}/all"

Record = Mapping[str, object]
ParsedSubjectRow = tuple[int, int, int, int, str, str, int, int, int, datetime, datetime]
JSONScalar = str | int | float | bool | None


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
    skipped_missing_name: int = 0
    skipped_non_target_bin: int = 0
    duplicate_rows_in_batches: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "raw_records": self.raw_records,
            "parsed_records": self.parsed_records,
            "inserted_records": self.inserted_records,
            "unchanged_records": self.unchanged_records,
            "skipped_missing_id": self.skipped_missing_id,
            "skipped_missing_name": self.skipped_missing_name,
            "skipped_non_target_bin": self.skipped_non_target_bin,
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
    return datetime.now(UTC)


def _datetime_to_iso(value: datetime) -> str:
    return value.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _iso_to_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_datetime(value: object, *, default: datetime) -> datetime:
    if value is None:
        return default

    if isinstance(value, datetime):
        if value.tzinfo is None:
            result = value.replace(tzinfo=UTC)
        else:
            result = value.astimezone(UTC)
        return _ensure_valid_datetime_range(result)

    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts = ts / 1000.0
        result = datetime.fromtimestamp(ts, tz=UTC)
        return _ensure_valid_datetime_range(result)

    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value).decode("utf-8", errors="ignore").strip()
    else:
        raw = str(value).strip()

    if not raw:
        return default

    try:
        parsed_naive = _iso_to_datetime(raw)
        result = parsed_naive.replace(tzinfo=UTC)
        return _ensure_valid_datetime_range(result)
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
            result = parsed.replace(tzinfo=UTC)
            return _ensure_valid_datetime_range(result)
        except ValueError:
            continue

    return default


def _ensure_valid_datetime_range(dt: datetime) -> datetime:
    EARLIEST_ALLOWED = datetime(1970, 1, 1, tzinfo=UTC)
    LATEST_ALLOWED = datetime(2100, 12, 31, 23, 59, 59, tzinfo=UTC)
    if dt < EARLIEST_ALLOWED:
        return EARLIEST_ALLOWED
    if dt > LATEST_ALLOWED:
        return LATEST_ALLOWED
    return dt


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


def _load_yaml(path: Path) -> object:
    yaml_module = importlib.import_module("yaml")
    safe_load = cast(Callable[[str], object], getattr(yaml_module, "safe_load"))
    return safe_load(path.read_text(encoding="utf-8"))


def _to_uint8(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        coerced = value
    elif isinstance(value, float):
        coerced = int(value)
    elif isinstance(value, (str, bytes, bytearray)):
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return default
    else:
        return default
    return max(0, min(255, coerced))


def _to_flag(value: object, default: int = 0) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "active"}:
            return 1
        if normalized in {"false", "no", "n", "0", "inactive"}:
            return 0
    return 1 if _to_uint8(value, default=default) > 0 else 0


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


def _first_non_empty_str(record: Record, keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _as_object_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return None


def _load_target_bins(config_path: Path) -> list[int]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    loaded_obj = _load_yaml(config_path)
    loaded = _as_object_mapping(loaded_obj)
    if loaded is None:
        raise ValueError("config.yaml must be a dictionary at top level")

    organizations = _as_object_mapping(loaded.get("organizations"))
    if organizations is None:
        raise ValueError("config.yaml missing organizations section")

    bins_raw_obj = organizations.get("bins")
    if not isinstance(bins_raw_obj, list):
        raise ValueError("config.yaml organizations.bins must be a list")

    bins_values = cast(list[object], bins_raw_obj)
    bins: list[int] = []
    for item in bins_values:
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
        payload_raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Invalid checkpoint file %s: %s", path, error)
        return CheckpointState(None, None, 0, 0, 0, 0)

    payload = _as_object_mapping(payload_raw)
    if payload is None:
        return CheckpointState(None, None, 0, 0, 0, 0)

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
    _ = path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _compare_columns() -> tuple[str, ...]:
    return (
        "pid",
        "bin",
        "iin",
        "name_ru",
        "name_kz",
        "is_customer",
        "is_organizer",
        "is_supplier",
        "register_date",
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
        "FROM subjects WHERE id IN %(ids)s GROUP BY id"
        ")"
    )
    result = clickhouse.query(query, {"ids": tuple(ids)})
    rows = result.result_rows
    existing: dict[int, tuple[object, ...]] = {}
    for row in rows:
        if len(row) < 2:
            continue
        existing[_to_uint64(row[0], default=0)] = tuple(row[1:])
    return existing


def _upsert_batch(
    clickhouse: ClickHouseClient, rows: Sequence[ParsedSubjectRow]
) -> tuple[int, int, int]:
    if not rows:
        return 0, 0, 0

    latest_by_id: dict[int, ParsedSubjectRow] = {}
    for row in rows:
        row_id = row[0]
        current = latest_by_id.get(row_id)
        if current is None or row[10] >= current[10]:
            latest_by_id[row_id] = row

    duplicate_count = len(rows) - len(latest_by_id)
    deduped_rows = list(latest_by_id.values())
    existing = _fetch_existing_latest_by_ids(clickhouse, [row[0] for row in deduped_rows])

    insert_rows: list[ParsedSubjectRow] = []
    unchanged = 0
    for row in deduped_rows:
        row_id = row[0]
        comparable = tuple(row[1:10])
        if row_id in existing and existing[row_id] == comparable:
            unchanged += 1
            continue
        insert_rows.append(row)

    if not insert_rows:
        return 0, duplicate_count, unchanged

    columns = (
        "id",
        "pid",
        "bin",
        "iin",
        "name_ru",
        "name_kz",
        "is_customer",
        "is_organizer",
        "is_supplier",
        "register_date",
        "updated_at",
    )
    clickhouse.insert(
        "subjects",
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
    base_params: dict[str, JSONScalar] = {"bin": str(bin_value)}
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


def _parse_subject_record(
    record: Record,
    target_bins: set[int],
    stats: DataQualityStats,
) -> tuple[ParsedSubjectRow, datetime] | None:
    stats.raw_records += 1

    row_id = _to_uint64(record.get("pid", record.get("id")), default=0)
    if row_id <= 0:
        stats.skipped_missing_id += 1
        return None

    bin_value = _to_uint64(record.get("bin"), default=0)
    iin_value = _to_uint64(record.get("iin"), default=0)
    if bin_value <= 0:
        bin_value = iin_value

    name_ru = _first_non_empty_str(record, ("name_ru", "nameRu", "name"))
    name_kz = _first_non_empty_str(record, ("name_kz", "nameKz"))
    if not (name_ru or name_kz):
        stats.skipped_missing_name += 1
        return None

    parsed_subject = Subject.model_validate(
        {
            "id": row_id,
            "bin": str(bin_value),
            "name_ru": name_ru or None,
            "name_kz": name_kz or None,
        }
    )
    parsed_id = _to_uint64(parsed_subject.id, default=row_id)
    parsed_bin = _to_uint64(parsed_subject.bin, default=bin_value)

    pid = _to_uint64(record.get("pid", record.get("parent_id")), default=0)
    now = _now_utc_naive()
    register_date = _parse_datetime(
        record.get("register_date", record.get("crdate", record.get("created_at"))),
        default=now,
    )
    updated_at = _parse_datetime(
        record.get(
            "last_modified",
            record.get("lastModified", record.get("updated_at", record.get("crdate"))),
        ),
        default=register_date,
    )

    row: ParsedSubjectRow = (
        parsed_id,
        pid,
        parsed_bin,
        iin_value,
        parsed_subject.name_ru or "",
        parsed_subject.name_kz or "",
        _to_flag(record.get("is_customer", record.get("customer")), default=0),
        _to_flag(record.get("is_organizer", record.get("organizer")), default=0),
        _to_flag(record.get("is_supplier", record.get("supplier")), default=0),
        register_date,
        updated_at,
    )
    stats.parsed_records += 1
    return row, updated_at


def load_subjects(
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

    endpoints = (SUBJECTS, SUBJECTS_ALL)
    bins_ordered = sorted(target_bins)
    stats = DataQualityStats()

    if force:
        LOGGER.info("Force mode enabled: loading all records for configured BINs")
        checkpoint.endpoint_index = 0
        checkpoint.bin_index = 0

    start_endpoint_index = checkpoint.endpoint_index if since is not None and not force else 0
    start_bin_index = checkpoint.bin_index if since is not None and not force else 0

    batch_rows: list[ParsedSubjectRow] = []
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
                    parsed = _parse_subject_record(item, target_bins, stats)
                    if parsed is None:
                        continue

                    row, modified_at = parsed
                    if max_seen is None or modified_at > max_seen:
                        max_seen = modified_at

                    batch_rows.append(row)
                    if len(batch_rows) >= batch_size:
                        inserted, duplicates, unchanged = _upsert_batch(clickhouse, batch_rows)
                        stats.inserted_records += inserted
                        stats.duplicate_rows_in_batches += duplicates
                        stats.unchanged_records += unchanged
                        checkpoint.processed_records = stats.parsed_records
                        checkpoint.inserted_records = stats.inserted_records
                        checkpoint.max_seen_modified = max_seen
                        _write_checkpoint(checkpoint_path, checkpoint)
                        LOGGER.info(
                            "Subjects progress: parsed=%d inserted=%d duplicates=%d unchanged=%d",
                            stats.parsed_records,
                            stats.inserted_records,
                            stats.duplicate_rows_in_batches,
                            stats.unchanged_records,
                        )
                        batch_rows = []

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
        LOGGER.error("Subjects ETL failed; checkpoint persisted for recovery: %s", error)
        raise

    quality = stats.as_dict()
    LOGGER.info("Subjects ETL finished with quality stats: %s", quality)
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
    return load_subjects(
        ows_client,
        clickhouse,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        force=force,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for OWS subjects endpoint")
    _force_action = parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore incremental marker and run full per-BIN fetch",
    )
    _batch_action = parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Insert batch size (default: 500)",
    )
    _config_action = parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml with organizations.bins",
    )
    _checkpoint_action = parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to checkpoint JSON file",
    )
    _parser_actions = (_force_action, _batch_action, _config_action, _checkpoint_action)
    _ = _parser_actions

    args = parser.parse_args()
    summary = main(
        force=cast(bool, args.force),
        batch_size=cast(int, args.batch_size),
        config_path=cast(Path, args.config),
        checkpoint_path=cast(Path, args.checkpoint),
    )
    LOGGER.info("Subjects ETL summary: %s", summary)
