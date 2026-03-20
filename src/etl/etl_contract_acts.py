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
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Protocol, cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.api import BaseOWSClient, OWSAPIError
from src.api.endpoints import CONTRACT_ACTS

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_CHECKPOINT_PATH = Path(".sisyphus/state/etl_contract_acts_checkpoint.json")
CHECKPOINT_REPLAY_WINDOW_SECONDS = 300
TARGET_YEARS = {2024, 2025, 2026}

Record = Mapping[str, object]
JSONScalar = str | int | float | bool | None
ParsedActRow = tuple[
    int,
    int,
    str,
    int,
    int,
    Decimal,
    datetime,
    datetime,
    int,
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
    duplicate_rows_in_batches: int = 0
    skipped_missing_id: int = 0
    skipped_missing_contract_id: int = 0
    skipped_missing_contract_link: int = 0
    skipped_non_target_bin: int = 0
    skipped_non_target_year: int = 0
    skipped_out_of_scope: int = 0
    skipped_invalid_amount: int = 0
    status_approved: int = 0
    status_pending: int = 0
    status_draft: int = 0
    status_other: int = 0
    missing_act_type: int = 0
    execution_pct_computed: int = 0
    execution_pct_missing_contract_sum: int = 0
    execution_pct_over_100: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "raw_records": self.raw_records,
            "parsed_records": self.parsed_records,
            "inserted_records": self.inserted_records,
            "unchanged_records": self.unchanged_records,
            "duplicate_rows_in_batches": self.duplicate_rows_in_batches,
            "skipped_missing_id": self.skipped_missing_id,
            "skipped_missing_contract_id": self.skipped_missing_contract_id,
            "skipped_missing_contract_link": self.skipped_missing_contract_link,
            "skipped_non_target_bin": self.skipped_non_target_bin,
            "skipped_non_target_year": self.skipped_non_target_year,
            "skipped_out_of_scope": self.skipped_out_of_scope,
            "skipped_invalid_amount": self.skipped_invalid_amount,
            "status_approved": self.status_approved,
            "status_pending": self.status_pending,
            "status_draft": self.status_draft,
            "status_other": self.status_other,
            "missing_act_type": self.missing_act_type,
            "execution_pct_computed": self.execution_pct_computed,
            "execution_pct_missing_contract_sum": self.execution_pct_missing_contract_sum,
            "execution_pct_over_100": self.execution_pct_over_100,
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
            return datetime.strptime(raw, pattern)
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


def _load_yaml(path: Path) -> object:
    yaml_module = importlib.import_module("yaml")
    safe_load = cast(Callable[[str], object], getattr(yaml_module, "safe_load"))
    return safe_load(path.read_text(encoding="utf-8"))


def _as_object_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return None


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


def _classify_status(status_id: int, status_text: str) -> str:
    normalized = status_text.strip().lower()
    if any(token in normalized for token in ("approve", "approved", "утверж", "одобрен")):
        return "approved"
    if any(token in normalized for token in ("draft", "чернов")):
        return "draft"
    if any(token in normalized for token in ("pending", "ожида", "на соглас", "process")):
        return "pending"

    if status_id in {2, 3, 4}:
        return "approved"
    if status_id in {0, 1}:
        return "pending"
    return "other"


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
        "contract_id",
        "act_number",
        "customer_bin",
        "supplier_bin",
        "act_sum",
        "act_date",
        "approve_date",
        "ref_act_status_id",
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
        "FROM contract_acts WHERE id IN %(ids)s GROUP BY id"
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
    clickhouse: ClickHouseClient, rows: Sequence[ParsedActRow]
) -> tuple[int, int, int]:
    if not rows:
        return 0, 0, 0

    latest_by_id: dict[int, ParsedActRow] = {}
    for row in rows:
        row_id = row[0]
        current = latest_by_id.get(row_id)
        if current is None or row[9] >= current[9]:
            latest_by_id[row_id] = row

    duplicate_count = len(rows) - len(latest_by_id)
    deduped_rows = list(latest_by_id.values())
    existing = _fetch_existing_latest_by_ids(clickhouse, [row[0] for row in deduped_rows])

    insert_rows: list[ParsedActRow] = []
    unchanged = 0
    for row in deduped_rows:
        row_id = row[0]
        comparable = tuple(row[1:9])
        if row_id in existing and existing[row_id] == comparable:
            unchanged += 1
            continue
        insert_rows.append(row)

    if not insert_rows:
        return 0, duplicate_count, unchanged

    columns = (
        "id",
        "contract_id",
        "act_number",
        "customer_bin",
        "supplier_bin",
        "act_sum",
        "act_date",
        "approve_date",
        "ref_act_status_id",
        "updated_at",
    )
    clickhouse.insert(
        "contract_acts",
        insert_rows,
        column_names=columns,
    )
    return len(insert_rows), duplicate_count, unchanged


def _fetch_contract_scope(
    clickhouse: ClickHouseClient,
    contract_ids: Sequence[int],
) -> dict[int, tuple[int, int, Decimal, int]]:
    if not contract_ids:
        return {}

    query = (
        "SELECT id, "
        "argMax(customer_bin, updated_at) AS customer_bin, "
        "argMax(supplier_bin, updated_at) AS supplier_bin, "
        "argMax(contract_sum, updated_at) AS contract_sum, "
        "argMax(sign_date, updated_at) AS sign_date "
        "FROM contracts "
        "WHERE id IN %(ids)s "
        "GROUP BY id"
    )
    result = clickhouse.query(query, {"ids": tuple(contract_ids)})
    rows = result.result_rows
    scope: dict[int, tuple[int, int, Decimal, int]] = {}
    for row in rows:
        if len(row) < 5:
            continue
        contract_id = _to_uint64(row[0], default=0)
        if contract_id <= 0:
            continue
        customer_bin = _to_uint64(row[1], default=0)
        supplier_bin = _to_uint64(row[2], default=0)
        contract_sum = _to_decimal(row[3], scale=2, default=Decimal("0.00"))
        sign_date = _parse_datetime(row[4], default=_now_utc_naive())
        scope[contract_id] = (
            customer_bin,
            supplier_bin,
            contract_sum,
            _to_uint16(sign_date.year, 0),
        )
    return scope


def _filter_rows_by_contract_scope(
    clickhouse: ClickHouseClient,
    rows: Sequence[ParsedActRow],
    target_bins: set[int],
    stats: DataQualityStats,
) -> list[ParsedActRow]:
    if not rows:
        return []

    contract_ids = sorted({row[1] for row in rows if row[1] > 0})
    contract_scope = _fetch_contract_scope(clickhouse, contract_ids)

    filtered: list[ParsedActRow] = []
    for row in rows:
        row_id = row[0]
        contract_id = row[1]
        act_number = row[2]
        customer_bin = row[3]
        supplier_bin = row[4]
        act_sum = row[5]
        act_date = row[6]
        approve_date = row[7]
        ref_act_status_id = row[8]
        updated_at = row[9]

        if contract_id not in contract_scope:
            stats.skipped_missing_contract_link += 1
            continue

        contract_customer_bin, contract_supplier_bin, contract_sum, contract_year = contract_scope[
            contract_id
        ]
        if contract_customer_bin not in target_bins:
            stats.skipped_non_target_bin += 1
            continue
        if contract_year not in TARGET_YEARS:
            stats.skipped_non_target_year += 1
            continue

        normalized_customer = customer_bin if customer_bin > 0 else contract_customer_bin
        normalized_supplier = supplier_bin if supplier_bin > 0 else contract_supplier_bin
        if normalized_customer != contract_customer_bin:
            normalized_customer = contract_customer_bin

        if contract_customer_bin not in target_bins or contract_year not in TARGET_YEARS:
            stats.skipped_out_of_scope += 1
            continue

        if contract_sum > Decimal("0.00"):
            execution_pct = (act_sum / contract_sum * Decimal("100")).quantize(
                Decimal("1.00"),
                rounding=ROUND_HALF_UP,
            )
            stats.execution_pct_computed += 1
            if execution_pct > Decimal("100.00"):
                stats.execution_pct_over_100 += 1
        else:
            stats.execution_pct_missing_contract_sum += 1

        filtered.append(
            (
                row_id,
                contract_id,
                act_number,
                normalized_customer,
                normalized_supplier,
                act_sum,
                act_date,
                approve_date,
                ref_act_status_id,
                updated_at,
            )
        )

    return filtered


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


def _parse_act_record(
    record: Record,
    target_bins: set[int],
    stats: DataQualityStats,
    fallback_customer_bin: int,
) -> tuple[ParsedActRow | None, datetime | None]:
    stats.raw_records += 1

    act_id = _to_uint64(
        record.get("id", record.get("act_id", record.get("contract_act_id"))), default=0
    )
    if act_id <= 0:
        stats.skipped_missing_id += 1
        return None, None

    contract_id = _to_uint64(
        record.get("contract_id", record.get("id_contract", record.get("contractId"))),
        default=0,
    )
    if contract_id <= 0:
        stats.skipped_missing_contract_id += 1
        return None, None

    customer_bin = _to_uint64(
        record.get("customer_bin", record.get("customerBin", record.get("bin"))),
        default=fallback_customer_bin,
    )
    if customer_bin <= 0:
        stats.skipped_non_target_bin += 1
        return None, None
    if target_bins and customer_bin not in target_bins:
        stats.skipped_non_target_bin += 1
        return None, None

    supplier_bin = _to_uint64(
        record.get("supplier_bin", record.get("supplierBin", record.get("supplier"))),
        default=0,
    )
    act_number = _first_non_empty_str(record, ("act_number", "number", "act_no", "num"))

    act_sum = _to_decimal(
        record.get("act_sum", record.get("sum", record.get("amount"))),
        scale=2,
        default=Decimal("0.00"),
    )
    if act_sum < Decimal("0.00"):
        stats.skipped_invalid_amount += 1
        return None, None

    now = _now_utc_naive()
    act_date = _parse_datetime(
        record.get("act_date", record.get("date_act", record.get("actDate"))),
        default=now,
    )
    approve_date = _parse_datetime(
        record.get("approve_date", record.get("approved_date", record.get("date_approve"))),
        default=act_date,
    )
    updated_at = _parse_datetime(
        record.get(
            "last_modified",
            record.get("lastModified", record.get("updated_at", record.get("crdate"))),
        ),
        default=approve_date,
    )

    ref_act_status_id = _to_uint64(
        record.get("ref_act_status_id", record.get("status_id", record.get("statusId"))),
        default=0,
    )
    status_text = _first_non_empty_str(
        record,
        (
            "act_status_name_ru",
            "act_status_name_kz",
            "status_name_ru",
            "status_name_kz",
            "status_name",
            "status",
        ),
    )
    status_group = _classify_status(ref_act_status_id, status_text)
    if status_group == "approved":
        stats.status_approved += 1
    elif status_group == "pending":
        stats.status_pending += 1
    elif status_group == "draft":
        stats.status_draft += 1
    else:
        stats.status_other += 1

    act_type = _first_non_empty_str(
        record, ("type_act", "act_type", "act_category", "type", "category")
    )
    if not act_type:
        stats.missing_act_type += 1

    row: ParsedActRow = (
        act_id,
        contract_id,
        act_number,
        customer_bin,
        supplier_bin,
        act_sum,
        act_date,
        approve_date,
        ref_act_status_id,
        updated_at,
    )
    stats.parsed_records += 1
    return row, updated_at


def load_contract_acts(
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

    endpoints = (CONTRACT_ACTS,)
    bins_ordered = sorted(target_bins)
    stats = DataQualityStats()

    if force:
        LOGGER.info("Force mode enabled: loading all contract act records for configured BINs")
        checkpoint.endpoint_index = 0
        checkpoint.bin_index = 0

    start_endpoint_index = checkpoint.endpoint_index if since is not None and not force else 0
    start_bin_index = checkpoint.bin_index if since is not None and not force else 0

    batch_rows: list[ParsedActRow] = []
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
                    row, modified_at = _parse_act_record(item, target_bins, stats, bin_value)
                    if modified_at is not None and (max_seen is None or modified_at > max_seen):
                        max_seen = modified_at

                    if row is None:
                        continue

                    batch_rows.append(row)
                    while len(batch_rows) >= batch_size:
                        current_batch = batch_rows[:batch_size]
                        valid_batch = _filter_rows_by_contract_scope(
                            clickhouse,
                            current_batch,
                            target_bins,
                            stats,
                        )
                        inserted, duplicates, unchanged = _upsert_batch(clickhouse, valid_batch)
                        stats.inserted_records += inserted
                        stats.duplicate_rows_in_batches += duplicates
                        stats.unchanged_records += unchanged
                        checkpoint.processed_records = stats.parsed_records
                        checkpoint.inserted_records = stats.inserted_records
                        checkpoint.max_seen_modified = max_seen
                        _write_checkpoint(checkpoint_path, checkpoint)
                        LOGGER.info(
                            "Contract acts progress: parsed=%d inserted=%d duplicates=%d unchanged=%d",
                            stats.parsed_records,
                            stats.inserted_records,
                            stats.duplicate_rows_in_batches,
                            stats.unchanged_records,
                        )
                        batch_rows = batch_rows[batch_size:]

        if batch_rows:
            valid_batch = _filter_rows_by_contract_scope(clickhouse, batch_rows, target_bins, stats)
            inserted, duplicates, unchanged = _upsert_batch(clickhouse, valid_batch)
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
        LOGGER.error("Contract acts ETL failed; checkpoint persisted for recovery: %s", error)
        raise

    quality = stats.as_dict()
    LOGGER.info(
        "Contract acts ETL finished with quality stats: %s. Status filtering logic: no exclusion; approved/pending/draft are all ingested and tracked.",
        quality,
    )
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
    return load_contract_acts(
        ows_client,
        clickhouse,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        force=force,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for OWS contract acts endpoint")
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
    LOGGER.info("Contract acts ETL summary: %s", summary)
