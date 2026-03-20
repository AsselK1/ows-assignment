from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.api import BaseOWSClient, JOURNAL, OWSAPIError
from src.etl import (
    load_announcements,
    load_contract_acts,
    load_contracts,
    load_lots,
    load_plans,
    load_subjects,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_STATE_PATH = Path(".sisyphus/state/scheduler_state.json")

JSONScalar = str | int | float | bool | None
EntityKey = str


class ClickHouseClient(Protocol):
    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]: ...

    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class EntitySpec:
    entity: EntityKey
    journal_entity: str
    loader: Callable[..., dict[str, int]]
    checkpoint_path: Path
    baseline_table: str
    baseline_column: str


ENTITY_SPECS: dict[EntityKey, EntitySpec] = {
    "subjects": EntitySpec(
        entity="subjects",
        journal_entity="subjects",
        loader=load_subjects,
        checkpoint_path=Path(".sisyphus/state/etl_subjects_checkpoint.json"),
        baseline_table="subjects",
        baseline_column="register_date",
    ),
    "plans": EntitySpec(
        entity="plans",
        journal_entity="plans",
        loader=load_plans,
        checkpoint_path=Path(".sisyphus/state/etl_plans_checkpoint.json"),
        baseline_table="plans",
        baseline_column="publish_date",
    ),
    "announcements": EntitySpec(
        entity="announcements",
        journal_entity="trd-buy",
        loader=load_announcements,
        checkpoint_path=Path(".sisyphus/state/etl_announcements_checkpoint.json"),
        baseline_table="announcements",
        baseline_column="publish_date",
    ),
    "lots": EntitySpec(
        entity="lots",
        journal_entity="lots",
        loader=load_lots,
        checkpoint_path=Path(".sisyphus/state/etl_lots_checkpoint.json"),
        baseline_table="lots",
        baseline_column="created_at",
    ),
    "contracts": EntitySpec(
        entity="contracts",
        journal_entity="contracts",
        loader=load_contracts,
        checkpoint_path=Path(".sisyphus/state/etl_contracts_checkpoint.json"),
        baseline_table="contracts",
        baseline_column="sign_date",
    ),
    "contract_acts": EntitySpec(
        entity="contract_acts",
        journal_entity="contract-acts",
        loader=load_contract_acts,
        checkpoint_path=Path(".sisyphus/state/etl_contract_acts_checkpoint.json"),
        baseline_table="contract_acts",
        baseline_column="act_date",
    ),
}

ENTITY_ORDER: tuple[EntityKey, ...] = (
    "subjects",
    "plans",
    "announcements",
    "lots",
    "contracts",
    "contract_acts",
)


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


def _to_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default
    return default


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


def _load_state(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        raw_payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Invalid scheduler state file %s: %s", path, error)
        return {}
    payload = _as_object_mapping(raw_payload)
    if payload is None:
        return {}

    state: dict[str, dict[str, object]] = {}
    for key, value in payload.items():
        if key not in ENTITY_SPECS:
            continue
        nested = _as_object_mapping(value)
        if nested is None:
            continue
        state[key] = dict(nested)
    return state


def _write_state(path: Path, state: Mapping[str, Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        entity: {
            "last_sync": value.get("last_sync", value.get("last_sync_timestamp")),
            "last_sync_timestamp": value.get("last_sync", value.get("last_sync_timestamp")),
            "last_success": value.get("last_success", value.get("last_success_timestamp")),
            "last_success_timestamp": value.get(
                "last_success",
                value.get("last_success_timestamp"),
            ),
            "failure_count": _to_int(value.get("failure_count", 0), default=0),
            "next_retry_delay_seconds": _to_int(
                value.get("next_retry_delay_seconds", 60),
                default=60,
            ),
            "next_retry_delay": _to_int(
                value.get("next_retry_delay_seconds", value.get("next_retry_delay", 60)),
                default=60,
            ),
            "next_retry_at": value.get("next_retry_at"),
            "last_error": value.get("last_error"),
        }
        for entity, value in state.items()
    }
    _ = path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_max_timestamp(
    clickhouse: ClickHouseClient,
    *,
    table: str,
    column: str,
) -> datetime | None:
    rows = clickhouse.execute(f"SELECT max({column}) FROM {table}")
    if not rows:
        return None
    value = rows[0][0]
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)
    if isinstance(value, str) and value.strip():
        try:
            return _iso_to_datetime(value)
        except ValueError:
            return None
    return None


def _initialize_state_if_needed(
    state: dict[str, dict[str, object]],
    *,
    clickhouse: ClickHouseClient | None,
    state_path: Path,
) -> dict[str, dict[str, object]]:
    changed = False
    now = _now_utc_naive()

    for entity in ENTITY_ORDER:
        if entity in state:
            continue
        spec = ENTITY_SPECS[entity]
        baseline: datetime | None
        if clickhouse is None:
            baseline = None
        else:
            try:
                baseline = _get_max_timestamp(
                    clickhouse,
                    table=spec.baseline_table,
                    column=spec.baseline_column,
                )
            except Exception as error:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to initialize baseline from %s.%s for %s: %s",
                    spec.baseline_table,
                    spec.baseline_column,
                    entity,
                    error,
                )
                baseline = None

        baseline_ts = baseline or (now - timedelta(hours=24))
        iso = _datetime_to_iso(baseline_ts)
        state[entity] = {
            "last_sync": iso,
            "last_success": iso,
            "failure_count": 0,
            "next_retry_delay_seconds": 60,
            "next_retry_at": None,
            "last_error": None,
        }
        changed = True

    if changed:
        _write_state(state_path, state)
    return state


def _extract_entry_id(entry: Mapping[str, object]) -> int:
    for key in ("entity_id", "id", "record_id", "object_id"):
        parsed = _to_uint64(entry.get(key), default=0)
        if parsed > 0:
            return parsed
    return 0


def _extract_entry_modified_at(entry: Mapping[str, object]) -> datetime | None:
    for key in ("modified_at", "updated_at", "created_at", "timestamp"):
        raw = entry.get(key)
        if isinstance(raw, str) and raw.strip():
            try:
                return _iso_to_datetime(raw)
            except ValueError:
                continue
    return None


def _fetch_journal_ids(
    ows_client: BaseOWSClient,
    *,
    spec: EntitySpec,
    from_date: datetime,
    to_date: datetime,
) -> tuple[list[int], datetime | None, bool]:
    params: dict[str, JSONScalar] = {
        "entity": spec.journal_entity,
        "from_date": _datetime_to_iso(from_date),
        "to_date": _datetime_to_iso(to_date),
    }
    changed_ids: set[int] = set()
    max_modified: datetime | None = None

    try:
        for item in ows_client.paginate(JOURNAL, params=params):
            entry = cast(Mapping[str, object], item)
            entity_id = _extract_entry_id(entry)
            if entity_id > 0:
                changed_ids.add(entity_id)
            modified_at = _extract_entry_modified_at(entry)
            if modified_at is not None and (max_modified is None or modified_at > max_modified):
                max_modified = modified_at
    except OWSAPIError as error:
        is_not_found = "404" in str(error)
        if is_not_found:
            LOGGER.warning(
                "Journal endpoint unavailable for %s (%s). Falling back to ETL incremental checkpoint path.",
                spec.entity,
                error,
            )
            return [], None, True
        raise

    return sorted(changed_ids), max_modified, False


def _loader_supports_id_filter(loader: Callable[..., dict[str, int]]) -> bool:
    signature = inspect.signature(loader)
    return "id_filter" in signature.parameters


def _next_backoff_seconds(failure_count: int) -> int:
    power = max(0, failure_count - 1)
    delay = 60 * (1 << power)
    return int(min(delay, 900))


def _entity_retry_not_due(entity_state: Mapping[str, object], now: datetime) -> bool:
    next_retry_at_raw = entity_state.get("next_retry_at")
    if not isinstance(next_retry_at_raw, str) or not next_retry_at_raw.strip():
        return False
    try:
        next_retry_at = _iso_to_datetime(next_retry_at_raw)
    except ValueError:
        return False
    return next_retry_at > now


def _refresh_entity(
    entity: EntityKey,
    *,
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient | None,
    state: dict[str, dict[str, object]],
    state_path: Path,
    config_path: Path,
    dry_run: bool,
) -> bool:
    now = _now_utc_naive()
    spec = ENTITY_SPECS[entity]
    entity_state = state[entity]

    if _entity_retry_not_due(entity_state, now):
        LOGGER.warning(
            "Skipping %s refresh due to backoff until %s",
            entity,
            entity_state.get("next_retry_at"),
        )
        _write_state(state_path, state)
        return False

    last_sync_raw = entity_state.get("last_sync")
    if last_sync_raw is None:
        last_sync_raw = entity_state.get("last_sync_timestamp")
    if isinstance(last_sync_raw, str) and last_sync_raw.strip():
        try:
            from_date = _iso_to_datetime(last_sync_raw)
        except ValueError:
            from_date = now - timedelta(hours=24)
    else:
        from_date = now - timedelta(hours=24)

    try:
        changed_ids, max_modified, used_fallback = _fetch_journal_ids(
            ows_client,
            spec=spec,
            from_date=from_date,
            to_date=now,
        )

        if dry_run:
            sample_ids = changed_ids[:10]
            LOGGER.info(
                "[DRY-RUN] Would sync %s: %d changed records (IDs: %s)",
                entity,
                len(changed_ids),
                sample_ids,
            )
        else:
            if clickhouse is None:
                raise RuntimeError(
                    "ClickHouse client is required for non-dry-run scheduler refresh"
                )
            if changed_ids and _loader_supports_id_filter(spec.loader):
                _ = spec.loader(
                    ows_client,
                    clickhouse,
                    config_path=config_path,
                    checkpoint_path=spec.checkpoint_path,
                    force=False,
                    id_filter=changed_ids,
                )
            else:
                if changed_ids and not used_fallback:
                    LOGGER.info(
                        "%s loader does not support id_filter; using checkpoint-based incremental refresh",
                        entity,
                    )
                _ = spec.loader(
                    ows_client,
                    clickhouse,
                    config_path=config_path,
                    checkpoint_path=spec.checkpoint_path,
                    force=False,
                )

        success_sync_ts = max_modified or now
        entity_state["last_sync"] = _datetime_to_iso(success_sync_ts)
        entity_state["last_success"] = _datetime_to_iso(now)
        entity_state["failure_count"] = 0
        entity_state["next_retry_delay_seconds"] = 60
        entity_state["next_retry_at"] = None
        entity_state["last_error"] = None
        _write_state(state_path, state)
        return True
    except Exception as error:  # noqa: BLE001
        failure_count = _to_int(entity_state.get("failure_count", 0), default=0) + 1
        delay_seconds = _next_backoff_seconds(failure_count)
        next_retry_at = now + timedelta(seconds=delay_seconds)

        entity_state["failure_count"] = failure_count
        entity_state["next_retry_delay_seconds"] = delay_seconds
        entity_state["next_retry_at"] = _datetime_to_iso(next_retry_at)
        entity_state["last_error"] = str(error)
        _write_state(state_path, state)

        LOGGER.warning("Entity %s sync failed: %s", entity, error)
        if failure_count >= 3:
            LOGGER.error("Entity %s has failed 3+ consecutive syncs", entity)
        return False


def refresh_subjects(
    *,
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient | None,
    state: dict[str, dict[str, object]],
    state_path: Path,
    config_path: Path,
    dry_run: bool,
) -> bool:
    return _refresh_entity(
        "subjects",
        ows_client=ows_client,
        clickhouse=clickhouse,
        state=state,
        state_path=state_path,
        config_path=config_path,
        dry_run=dry_run,
    )


def refresh_plans(
    *,
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient | None,
    state: dict[str, dict[str, object]],
    state_path: Path,
    config_path: Path,
    dry_run: bool,
) -> bool:
    return _refresh_entity(
        "plans",
        ows_client=ows_client,
        clickhouse=clickhouse,
        state=state,
        state_path=state_path,
        config_path=config_path,
        dry_run=dry_run,
    )


def refresh_announcements(
    *,
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient | None,
    state: dict[str, dict[str, object]],
    state_path: Path,
    config_path: Path,
    dry_run: bool,
) -> bool:
    return _refresh_entity(
        "announcements",
        ows_client=ows_client,
        clickhouse=clickhouse,
        state=state,
        state_path=state_path,
        config_path=config_path,
        dry_run=dry_run,
    )


def refresh_lots(
    *,
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient | None,
    state: dict[str, dict[str, object]],
    state_path: Path,
    config_path: Path,
    dry_run: bool,
) -> bool:
    return _refresh_entity(
        "lots",
        ows_client=ows_client,
        clickhouse=clickhouse,
        state=state,
        state_path=state_path,
        config_path=config_path,
        dry_run=dry_run,
    )


def refresh_contracts(
    *,
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient | None,
    state: dict[str, dict[str, object]],
    state_path: Path,
    config_path: Path,
    dry_run: bool,
) -> bool:
    return _refresh_entity(
        "contracts",
        ows_client=ows_client,
        clickhouse=clickhouse,
        state=state,
        state_path=state_path,
        config_path=config_path,
        dry_run=dry_run,
    )


def refresh_contract_acts(
    *,
    ows_client: BaseOWSClient,
    clickhouse: ClickHouseClient | None,
    state: dict[str, dict[str, object]],
    state_path: Path,
    config_path: Path,
    dry_run: bool,
) -> bool:
    return _refresh_entity(
        "contract_acts",
        ows_client=ows_client,
        clickhouse=clickhouse,
        state=state,
        state_path=state_path,
        config_path=config_path,
        dry_run=dry_run,
    )


ENTITY_REFRESHERS: dict[EntityKey, Callable[..., bool]] = {
    "subjects": refresh_subjects,
    "plans": refresh_plans,
    "announcements": refresh_announcements,
    "lots": refresh_lots,
    "contracts": refresh_contracts,
    "contract_acts": refresh_contract_acts,
}


def refresh_all_entities(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    dry_run: bool = False,
    entity: EntityKey | None = None,
) -> dict[str, bool]:
    ows_client = BaseOWSClient()
    clickhouse: ClickHouseClient | None
    try:
        clickhouse = _get_clickhouse_client()
    except Exception as error:  # noqa: BLE001
        if not dry_run:
            raise
        LOGGER.warning(
            "ClickHouse unavailable for dry-run; using fallback state baselines: %s", error
        )
        clickhouse = None

    try:
        state = _load_state(state_path)
        state = _initialize_state_if_needed(state, clickhouse=clickhouse, state_path=state_path)

        selected_entities: Sequence[EntityKey]
        if entity is not None:
            selected_entities = (entity,)
        else:
            selected_entities = ENTITY_ORDER

        LOGGER.info(
            "Running scheduler refresh in sequential mode (target cadence: every 6 hours / 4x daily)"
        )

        results: dict[str, bool] = {}
        for current_entity in selected_entities:
            refresher = ENTITY_REFRESHERS[current_entity]
            result = refresher(
                ows_client=ows_client,
                clickhouse=clickhouse,
                state=state,
                state_path=state_path,
                config_path=config_path,
                dry_run=dry_run,
            )
            results[current_entity] = result
        return results
    finally:
        if clickhouse is not None:
            clickhouse.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh scheduler for OWS entities using journal-based incremental sync. "
            "Run this command every 6 hours via an external scheduler."
        )
    )
    _config_action = parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml with organizations.bins",
    )
    _dry_run_action = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and log sync plan without executing ETL writes",
    )
    _entity_action = parser.add_argument(
        "--entity",
        choices=list(ENTITY_ORDER),
        default=None,
        help="Optional single-entity refresh selector",
    )
    _ = (_config_action, _dry_run_action, _entity_action)
    return parser.parse_args()


def main() -> dict[str, bool]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    return refresh_all_entities(
        config_path=cast(Path, args.config),
        dry_run=cast(bool, args.dry_run),
        entity=cast(EntityKey | None, args.entity),
    )


if __name__ == "__main__":
    summary = main()
    LOGGER.info("Scheduler refresh summary: %s", summary)
