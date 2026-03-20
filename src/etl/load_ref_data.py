from __future__ import annotations

import argparse
import hashlib
import importlib
import logging
import os
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar, cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.api import BaseOWSClient, OWSAPIError, REF_ENSTR, REF_KATO, REF_MKEI

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500
T = TypeVar("T")
Record = Mapping[str, object]
ParsedEnstrRow = tuple[str, str, str, int, str, int, int, datetime]
ParsedKatoRow = tuple[str, str, str, int, str, int, int, datetime]
ParsedMkeiRow = tuple[str, str, str, str, int, int, datetime]
ParsedRow = ParsedEnstrRow | ParsedKatoRow | ParsedMkeiRow


class ClickHouseClient(Protocol):
    def query(self, query: str, parameters: dict | None = None) -> Any: ...

    def command(self, query: str, parameters: dict | list | None = None) -> None: ...

    def insert(self, table: str, data: Any, column_names: Sequence[str] | None = None) -> None: ...


class RecordParser(Protocol):
    def __call__(self, item: Record) -> ParsedRow | None: ...


@dataclass
class DataQualityStats:
    api_records: int = 0
    parsed_records: int = 0
    malformed_records: int = 0
    duplicate_keys_in_batch: int = 0
    unchanged_records: int = 0


@dataclass
class LoadResult:
    inserted_records: int = 0
    stats: DataQualityStats = field(default_factory=DataQualityStats)


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


def _chunked(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _first_present_str(record: Record, keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


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


def _to_active_flag(value: object, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "active"}:
            return 1
        if normalized in {"false", "no", "n", "0", "inactive"}:
            return 0
    return 1 if _to_uint8(value, default=default) > 0 else 0


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
        return int(text) if text else default
    if isinstance(value, bytearray):
        text = bytes(value).decode("utf-8", errors="ignore").strip()
        return int(text) if text else default
    if isinstance(value, str):
        text = value.strip()
        return int(text) if text else default
    return default


def _hash_row(parts: Sequence[object]) -> int:
    normalized = "|".join(str(part).strip() for part in parts)
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _parse_enstr(item: Record) -> ParsedEnstrRow | None:
    code = _first_present_str(item, ("code", "enstr_code"))
    name_ru = _first_present_str(item, ("name_ru", "ru", "name"))
    name_kz = _first_present_str(item, ("name_kz", "kz"))
    if not code or not name_ru or not name_kz:
        LOGGER.warning("Skipping malformed ENSTR record (required fields missing): %s", item)
        return None

    level = _to_uint8(item.get("level", item.get("lvl", item.get("ref_level"))), default=0)
    parent_code = _first_present_str(item, ("parent_code", "parent", "parentCode"))
    is_active = _to_active_flag(
        item.get("is_active", item.get("active", item.get("status"))), default=1
    )
    row_hash = _hash_row((code, name_ru, name_kz, level, parent_code, is_active))
    return (code, name_ru, name_kz, level, parent_code, is_active, row_hash, _now_utc())


def _parse_kato(item: Record) -> ParsedKatoRow | None:
    code = _first_present_str(item, ("code", "kato_code"))
    name_ru = _first_present_str(item, ("name_ru", "ru", "name"))
    name_kz = _first_present_str(item, ("name_kz", "kz"))
    if not code or not name_ru or not name_kz:
        LOGGER.warning("Skipping malformed KATO record (required fields missing): %s", item)
        return None

    level = _to_uint8(item.get("level", item.get("lvl", item.get("ref_level"))), default=0)
    parent_code = _first_present_str(item, ("parent_code", "parent", "parentCode"))
    is_active = _to_active_flag(
        item.get("is_active", item.get("active", item.get("status"))), default=1
    )
    row_hash = _hash_row((code, name_ru, name_kz, level, parent_code, is_active))
    return (code, name_ru, name_kz, level, parent_code, is_active, row_hash, _now_utc())


def _parse_mkei(item: Record) -> ParsedMkeiRow | None:
    code = _first_present_str(item, ("code", "mkei_code"))
    name_ru = _first_present_str(item, ("name_ru", "ru", "name"))
    name_kz = _first_present_str(item, ("name_kz", "kz"))
    if not code or not name_ru or not name_kz:
        LOGGER.warning("Skipping malformed MKEI record (required fields missing): %s", item)
        return None

    short_name = _first_present_str(item, ("short_name", "shortName", "abbr", "brief"))
    is_active = _to_active_flag(
        item.get("is_active", item.get("active", item.get("status"))), default=1
    )
    row_hash = _hash_row((code, name_ru, name_kz, short_name, is_active))
    return (code, name_ru, name_kz, short_name, is_active, row_hash, _now_utc())


def _table_has_column(clickhouse: ClickHouseClient, table: str, column: str) -> bool:
    result = clickhouse.query(
        (
            "SELECT count() FROM system.columns WHERE database = currentDatabase() "
            "AND table = %(table)s AND name = %(column)s"
        ),
        {"table": table, "column": column},
    )
    rows = result.result_rows
    if not rows:
        return False
    return _coerce_int(rows[0][0], default=0) > 0


def _ensure_hash_columns(clickhouse: ClickHouseClient) -> None:
    targets = (
        "reference_enstr",
        "reference_kato",
        "reference_mkei",
    )
    for table in targets:
        if _table_has_column(clickhouse, table, "row_hash"):
            continue
        clickhouse.command(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS row_hash UInt64 DEFAULT 0 AFTER is_active"
        )
        LOGGER.info("Added row_hash column to %s", table)


def _fetch_existing_latest(
    clickhouse: ClickHouseClient,
    table: str,
    key_column: str,
    value_columns: Sequence[str],
    keys: Sequence[str],
) -> dict[str, tuple[object, ...]]:
    if not keys:
        return {}

    argmax_columns = ", ".join(
        [f"argMax({column}, updated_at) AS {column}" for column in value_columns]
    )
    selected_columns = ", ".join([key_column, *value_columns])
    query = (
        f"SELECT {selected_columns} "
        f"FROM (SELECT {key_column}, {argmax_columns} FROM {table} "
        f"WHERE {key_column} IN %(keys)s GROUP BY {key_column})"
    )

    result = clickhouse.query(query, {"keys": tuple(keys)})
    rows = result.result_rows
    existing: dict[str, tuple[object, ...]] = {}
    for row in rows:
        if not row:
            continue
        key = str(row[0])
        existing[key] = tuple(row[1:])
    return existing


def _upsert_batch(
    clickhouse: ClickHouseClient,
    table: str,
    columns: Sequence[str],
    key_index: int,
    compare_indexes: Sequence[int],
    rows: Sequence[ParsedRow],
    force: bool,
) -> tuple[int, int, int]:
    if not rows:
        return 0, 0, 0

    latest_by_key: dict[str, ParsedRow] = {}
    for row in rows:
        latest_by_key[str(row[key_index])] = row
    duplicate_count = len(rows) - len(latest_by_key)
    deduped_rows = list(latest_by_key.values())

    insert_rows: list[ParsedRow] = deduped_rows
    unchanged = 0
    if not force:
        keys = [str(row[key_index]) for row in deduped_rows]
        value_columns = [columns[index] for index in compare_indexes]
        existing_map = _fetch_existing_latest(
            clickhouse=clickhouse,
            table=table,
            key_column=columns[key_index],
            value_columns=value_columns,
            keys=keys,
        )

        insert_rows = []
        for row in deduped_rows:
            key = str(row[key_index])
            candidate = tuple(row[index] for index in compare_indexes)
            if key not in existing_map:
                insert_rows.append(row)
                continue
            if existing_map[key] == candidate:
                unchanged += 1
                continue
            insert_rows.append(row)

    if not insert_rows:
        return 0, duplicate_count, unchanged

    clickhouse.insert(table, insert_rows, column_names=columns)
    return len(insert_rows), duplicate_count, unchanged


def _parsed_rows(
    name: str,
    items: Iterable[Record],
    parser: RecordParser,
    stats: DataQualityStats,
) -> Iterator[ParsedRow]:
    for item in items:
        stats.api_records += 1
        try:
            parsed = parser(item)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Error processing %s record: %s; item=%s", name, error, item)
            stats.malformed_records += 1
            continue
        if parsed is None:
            stats.malformed_records += 1
            continue
        stats.parsed_records += 1
        yield parsed


def _load_reference(
    name: str,
    endpoint: str,
    table: str,
    columns: Sequence[str],
    key_index: int,
    compare_indexes: Sequence[int],
    parser: RecordParser,
    *,
    force: bool,
    batch_size: int,
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient,
) -> LoadResult:
    LOGGER.info("Loading %s...", name)

    if force:
        clickhouse.command(f"TRUNCATE TABLE {table}")
        LOGGER.info("%s force mode enabled: table %s truncated", name, table)

    result = LoadResult()
    parsed_iter = _parsed_rows(
        name=name,
        items=ows_client.paginate(endpoint),
        parser=parser,
        stats=result.stats,
    )
    for batch in _chunked(parsed_iter, batch_size=batch_size):
        inserted, duplicate_count, unchanged = _upsert_batch(
            clickhouse=clickhouse,
            table=table,
            columns=columns,
            key_index=key_index,
            compare_indexes=compare_indexes,
            rows=batch,
            force=force,
        )
        result.inserted_records += inserted
        result.stats.duplicate_keys_in_batch += duplicate_count
        result.stats.unchanged_records += unchanged
        LOGGER.info("Loading %s... %d records loaded", name, result.inserted_records)

    LOGGER.info(
        "%s completed. api=%d parsed=%d inserted=%d malformed=%d duplicates=%d unchanged=%d",
        name,
        result.stats.api_records,
        result.stats.parsed_records,
        result.inserted_records,
        result.stats.malformed_records,
        result.stats.duplicate_keys_in_batch,
        result.stats.unchanged_records,
    )
    return result


def _get_table_count(clickhouse: ClickHouseClient, table: str) -> int:
    result = clickhouse.query(f"SELECT count() FROM {table}")
    rows = result.result_rows
    if not rows:
        return 0
    return _coerce_int(rows[0][0], default=0)


def load_enstr(
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient,
    *,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> LoadResult:
    return _load_reference(
        name="ENSTR",
        endpoint=REF_ENSTR,
        table="reference_enstr",
        columns=(
            "enstr_code",
            "name_ru",
            "name_kz",
            "level",
            "parent_code",
            "is_active",
            "row_hash",
            "updated_at",
        ),
        key_index=0,
        compare_indexes=(6,),
        parser=_parse_enstr,
        force=force,
        batch_size=batch_size,
        ows_client=ows_client,
        clickhouse=clickhouse,
    )


def load_kato(
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient,
    *,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> LoadResult:
    return _load_reference(
        name="KATO",
        endpoint=REF_KATO,
        table="reference_kato",
        columns=(
            "kato_code",
            "name_ru",
            "name_kz",
            "level",
            "parent_code",
            "is_active",
            "row_hash",
            "updated_at",
        ),
        key_index=0,
        compare_indexes=(6,),
        parser=_parse_kato,
        force=force,
        batch_size=batch_size,
        ows_client=ows_client,
        clickhouse=clickhouse,
    )


def load_mkei(
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient,
    *,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> LoadResult:
    return _load_reference(
        name="MKEI",
        endpoint=REF_MKEI,
        table="reference_mkei",
        columns=(
            "mkei_code",
            "name_ru",
            "name_kz",
            "short_name",
            "is_active",
            "row_hash",
            "updated_at",
        ),
        key_index=0,
        compare_indexes=(5,),
        parser=_parse_mkei,
        force=force,
        batch_size=batch_size,
        ows_client=ows_client,
        clickhouse=clickhouse,
    )


def main(force: bool = False, batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, int]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    ows_client = BaseOWSClient()
    clickhouse = _get_clickhouse_client()

    try:
        _ensure_hash_columns(clickhouse)
        enstr = load_enstr(ows_client, clickhouse, force=force, batch_size=batch_size)
        kato = load_kato(ows_client, clickhouse, force=force, batch_size=batch_size)
        mkei = load_mkei(ows_client, clickhouse, force=force, batch_size=batch_size)

        counts = {
            "reference_enstr": _get_table_count(clickhouse, "reference_enstr"),
            "reference_kato": _get_table_count(clickhouse, "reference_kato"),
            "reference_mkei": _get_table_count(clickhouse, "reference_mkei"),
        }
        if force:
            if counts["reference_enstr"] != enstr.stats.parsed_records:
                LOGGER.warning(
                    "ENSTR count mismatch: table=%d parsed=%d",
                    counts["reference_enstr"],
                    enstr.stats.parsed_records,
                )
            if counts["reference_kato"] != kato.stats.parsed_records:
                LOGGER.warning(
                    "KATO count mismatch: table=%d parsed=%d",
                    counts["reference_kato"],
                    kato.stats.parsed_records,
                )
            if counts["reference_mkei"] != mkei.stats.parsed_records:
                LOGGER.warning(
                    "MKEI count mismatch: table=%d parsed=%d",
                    counts["reference_mkei"],
                    mkei.stats.parsed_records,
                )
    except OWSAPIError as error:
        LOGGER.error("Reference data loading failed due to OWS API error: %s", error)
        raise

    summary = {
        "enstr_inserted": enstr.inserted_records,
        "kato_inserted": kato.inserted_records,
        "mkei_inserted": mkei.inserted_records,
        "enstr_api_records": enstr.stats.api_records,
        "kato_api_records": kato.stats.api_records,
        "mkei_api_records": mkei.stats.api_records,
    }
    LOGGER.info("Reference data loading complete: %s", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load ENSTR/KATO/MKEI reference data from OWS API")
    _force_action = parser.add_argument(
        "--force",
        action="store_true",
        help="Truncate reference tables before loading",
    )
    _batch_action = parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for inserts (default: 500)",
    )
    _ = (_force_action, _batch_action)
    cli_args = parser.parse_args()
    summary = main(force=cast(bool, cli_args.force), batch_size=cast(int, cli_args.batch_size))
    LOGGER.info("Reference data loading summary: %s", summary)
