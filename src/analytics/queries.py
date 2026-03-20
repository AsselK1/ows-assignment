from __future__ import annotations

import argparse
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

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_MIN_SAMPLE_SIZE = 5
MONEY_QUANT = Decimal("1.00")
SHARE_QUANT = Decimal("1.000000")


class ClickHouseClient(Protocol):
    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]: ...

    def disconnect(self) -> None: ...


SpendByBinRow = tuple[int, Decimal, int]
SpendByEnstrRow = tuple[str, Decimal, int]
SpendByRegionRow = tuple[str, str, Decimal, int]
SupplierConcentrationRow = tuple[int, Decimal, int, Decimal]
YearOverYearRow = tuple[int, Decimal, int, Decimal | None]


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


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Datetime value must not be empty")
    for pattern in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(normalized, pattern)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid datetime value: {value}") from error
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)


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


def _normalized_uint_values(values: Sequence[int] | None) -> tuple[int, ...]:
    if values is None:
        return tuple()
    normalized = tuple(sorted({v for v in values if v > 0}))
    return normalized


def _normalized_code_values(values: Sequence[str] | None, *, length: int) -> tuple[str, ...]:
    if values is None:
        return tuple()
    cleaned: set[str] = set()
    for value in values:
        digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
        if len(digits) >= length:
            cleaned.add(digits[:length])
    return tuple(sorted(cleaned))


def _build_contract_filters(
    *,
    bins: Sequence[int] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    enstr_codes: Sequence[str] | None = None,
    kato_codes: Sequence[str] | None = None,
) -> tuple[str, dict[str, object]]:
    conditions: list[str] = ["contract_sum > 0"]
    params: dict[str, object] = {}

    normalized_bins = _normalized_uint_values(bins)
    if normalized_bins:
        conditions.append("customer_bin IN %(bins)s")
        params["bins"] = normalized_bins

    if start_date is not None:
        conditions.append("sign_date >= %(start_date)s")
        params["start_date"] = start_date

    if end_date is not None:
        conditions.append("sign_date <= %(end_date)s")
        params["end_date"] = end_date

    normalized_enstr = _normalized_code_values(enstr_codes, length=4)
    if normalized_enstr:
        conditions.append("enstr_code_lvl4 IN %(enstr_codes)s")
        params["enstr_codes"] = normalized_enstr

    normalized_kato = _normalized_code_values(kato_codes, length=2)
    if normalized_kato:
        conditions.append("substring(kato_code, 1, 2) IN %(kato_prefixes)s")
        params["kato_prefixes"] = normalized_kato

    return " AND ".join(conditions), params


def _latest_contracts_cte(where_clause: str) -> str:
    return f"""
    WITH latest_contracts AS
    (
        SELECT
            id,
            argMax(customer_bin, updated_at) AS customer_bin,
            argMax(supplier_bin, updated_at) AS supplier_bin,
            argMax(enstr_code_lvl4, updated_at) AS enstr_code_lvl4,
            argMax(kato_code, updated_at) AS kato_code,
            argMax(contract_sum, updated_at) AS contract_sum,
            argMax(sign_date, updated_at) AS sign_date
        FROM contracts
        GROUP BY id
    ),
    filtered_contracts AS
    (
        SELECT
            id,
            customer_bin,
            supplier_bin,
            enstr_code_lvl4,
            kato_code,
            contract_sum,
            sign_date
        FROM latest_contracts
        WHERE {where_clause}
    )
    """


def total_spend_by_bin(
    clickhouse_client: ClickHouseClient,
    *,
    bins: Sequence[int] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    enstr_codes: Sequence[str] | None = None,
    kato_codes: Sequence[str] | None = None,
) -> list[SpendByBinRow]:
    where_clause, params = _build_contract_filters(
        bins=bins,
        start_date=start_date,
        end_date=end_date,
        enstr_codes=enstr_codes,
        kato_codes=kato_codes,
    )
    query = (
        _latest_contracts_cte(where_clause)
        + """
    SELECT
        customer_bin,
        sum(contract_sum) AS total_contract_sum,
        count() AS contract_count
    FROM filtered_contracts
    GROUP BY customer_bin
    ORDER BY total_contract_sum DESC, customer_bin ASC
    """
    )
    rows = clickhouse_client.execute(query, params)
    result: list[SpendByBinRow] = []
    for row in rows:
        if len(row) != 3:
            continue
        customer_bin = _to_uint64(row[0], default=0)
        if customer_bin <= 0:
            continue
        total_spend = _to_decimal(row[1], scale=2, default=Decimal("0.00"))
        contract_count = _to_uint64(row[2], default=0)
        result.append((customer_bin, total_spend, contract_count))
    return result


def spend_by_enstr(
    clickhouse_client: ClickHouseClient,
    *,
    bins: Sequence[int] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    enstr_codes: Sequence[str] | None = None,
    kato_codes: Sequence[str] | None = None,
) -> list[SpendByEnstrRow]:
    where_clause, params = _build_contract_filters(
        bins=bins,
        start_date=start_date,
        end_date=end_date,
        enstr_codes=enstr_codes,
        kato_codes=kato_codes,
    )
    query = (
        _latest_contracts_cte(where_clause)
        + """
    SELECT
        enstr_code_lvl4,
        sum(contract_sum) AS total_spend,
        count() AS contract_count
    FROM filtered_contracts
    WHERE length(enstr_code_lvl4) = 4
    GROUP BY enstr_code_lvl4
    ORDER BY total_spend DESC, enstr_code_lvl4 ASC
    """
    )
    rows = clickhouse_client.execute(query, params)
    result: list[SpendByEnstrRow] = []
    for row in rows:
        if len(row) != 3:
            continue
        enstr_code = "".join(ch for ch in str(row[0]).strip() if ch.isdigit())[:4]
        if len(enstr_code) < 4:
            continue
        total_spend = _to_decimal(row[1], scale=2, default=Decimal("0.00"))
        contract_count = _to_uint64(row[2], default=0)
        result.append((enstr_code, total_spend, contract_count))
    return result


def spend_by_region(
    clickhouse_client: ClickHouseClient,
    *,
    bins: Sequence[int] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    enstr_codes: Sequence[str] | None = None,
    kato_codes: Sequence[str] | None = None,
) -> list[SpendByRegionRow]:
    where_clause, params = _build_contract_filters(
        bins=bins,
        start_date=start_date,
        end_date=end_date,
        enstr_codes=enstr_codes,
        kato_codes=kato_codes,
    )
    query = (
        _latest_contracts_cte(where_clause)
        + """
    , latest_kato AS
    (
        SELECT
            kato_code,
            argMax(name_ru, updated_at) AS name_ru
        FROM reference_kato
        GROUP BY kato_code
    ),
    region_names AS
    (
        SELECT
            substring(kato_code, 1, 2) AS kato_prefix,
            any(name_ru) AS region_name
        FROM latest_kato
        WHERE length(kato_code) >= 2
        GROUP BY kato_prefix
    )
    SELECT
        substring(fc.kato_code, 1, 2) AS kato_code,
        ifNull(rn.region_name, 'Unknown region') AS region_name,
        sum(fc.contract_sum) AS total_spend,
        count() AS contract_count
    FROM filtered_contracts AS fc
    LEFT JOIN region_names AS rn
        ON rn.kato_prefix = substring(fc.kato_code, 1, 2)
    WHERE length(fc.kato_code) >= 2
    GROUP BY kato_code, region_name
    ORDER BY total_spend DESC, kato_code ASC
    """
    )
    rows = clickhouse_client.execute(query, params)
    result: list[SpendByRegionRow] = []
    for row in rows:
        if len(row) != 4:
            continue
        kato_code = "".join(ch for ch in str(row[0]).strip() if ch.isdigit())[:2]
        if len(kato_code) < 2:
            continue
        region_name = str(row[1]).strip() if row[1] is not None else "Unknown region"
        total_spend = _to_decimal(row[2], scale=2, default=Decimal("0.00"))
        contract_count = _to_uint64(row[3], default=0)
        result.append((kato_code, region_name, total_spend, contract_count))
    return result


def supplier_concentration(
    clickhouse_client: ClickHouseClient,
    *,
    bins: Sequence[int] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    enstr_codes: Sequence[str] | None = None,
    kato_codes: Sequence[str] | None = None,
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
) -> list[SupplierConcentrationRow]:
    where_clause, params = _build_contract_filters(
        bins=bins,
        start_date=start_date,
        end_date=end_date,
        enstr_codes=enstr_codes,
        kato_codes=kato_codes,
    )
    effective_min_sample_size = max(1, min_sample_size)
    params["min_sample_size"] = effective_min_sample_size
    query = (
        _latest_contracts_cte(where_clause)
        + """
    , customer_totals AS
    (
        SELECT
            customer_bin,
            sum(contract_sum) AS total_spend,
            count() AS contract_count
        FROM filtered_contracts
        GROUP BY customer_bin
        HAVING contract_count >= %(min_sample_size)s
    ),
    supplier_totals AS
    (
        SELECT
            customer_bin,
            supplier_bin,
            sum(contract_sum) AS supplier_spend
        FROM filtered_contracts
        GROUP BY customer_bin, supplier_bin
    )
    SELECT
        st.customer_bin,
        st.supplier_bin,
        st.supplier_spend,
        ct.total_spend,
        ct.contract_count
    FROM supplier_totals AS st
    INNER JOIN customer_totals AS ct
        ON st.customer_bin = ct.customer_bin
    WHERE st.supplier_bin > 0
    ORDER BY st.customer_bin ASC, st.supplier_spend DESC
    """
    )
    rows = clickhouse_client.execute(query, params)

    grouped: dict[int, list[tuple[Decimal, Decimal, int]]] = defaultdict(list)
    for row in rows:
        if len(row) != 5:
            continue
        customer_bin = _to_uint64(row[0], default=0)
        if customer_bin <= 0:
            continue
        supplier_spend = _to_decimal(row[2], scale=6, default=Decimal("0.000000"))
        total_spend = _to_decimal(row[3], scale=6, default=Decimal("0.000000"))
        contract_count = _to_uint64(row[4], default=0)
        if supplier_spend <= 0 or total_spend <= 0:
            continue
        grouped[customer_bin].append((supplier_spend, total_spend, contract_count))

    result: list[SupplierConcentrationRow] = []
    for customer_bin, supplier_rows in grouped.items():
        if not supplier_rows:
            continue
        total_spend = supplier_rows[0][1]
        if total_spend <= 0:
            continue

        hhi_score = Decimal("0.000000")
        top_supplier_share = Decimal("0.000000")
        for supplier_spend, _, _ in supplier_rows:
            share = (supplier_spend / total_spend).quantize(SHARE_QUANT, rounding=ROUND_HALF_UP)
            hhi_score += share * share
            if share > top_supplier_share:
                top_supplier_share = share

        result.append(
            (
                customer_bin,
                hhi_score.quantize(SHARE_QUANT, rounding=ROUND_HALF_UP),
                len(supplier_rows),
                top_supplier_share.quantize(SHARE_QUANT, rounding=ROUND_HALF_UP),
            )
        )

    result.sort(key=lambda row: row[1], reverse=True)
    return result


def year_over_year_trends(
    clickhouse_client: ClickHouseClient,
    *,
    bins: Sequence[int] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    enstr_codes: Sequence[str] | None = None,
    kato_codes: Sequence[str] | None = None,
) -> list[YearOverYearRow]:
    # Extend lookback by 1 year so the first requested year can have a
    # YoY comparison against the prior year's data.
    original_start_year: int | None = start_date.year if start_date is not None else None
    lookback_start: datetime | None = None
    if start_date is not None:
        lookback_start = start_date.replace(year=start_date.year - 1)
    where_clause, params = _build_contract_filters(
        bins=bins,
        start_date=lookback_start if lookback_start is not None else start_date,
        end_date=end_date,
        enstr_codes=enstr_codes,
        kato_codes=kato_codes,
    )
    query = (
        _latest_contracts_cte(where_clause)
        + """
    SELECT
        toYear(sign_date) AS spend_year,
        sum(contract_sum) AS total_spend,
        count() AS contract_count
    FROM filtered_contracts
    GROUP BY spend_year
    ORDER BY spend_year ASC
    """
    )
    rows = clickhouse_client.execute(query, params)
    raw_rows: list[tuple[int, Decimal, int]] = []
    for row in rows:
        if len(row) != 3:
            continue
        spend_year = _to_uint64(row[0], default=0)
        total_spend = _to_decimal(row[1], scale=2, default=Decimal("0.00"))
        contract_count = _to_uint64(row[2], default=0)
        if spend_year <= 0:
            continue
        raw_rows.append((spend_year, total_spend, contract_count))

    result: list[YearOverYearRow] = []
    previous_total: Decimal | None = None
    for spend_year, total_spend, contract_count in raw_rows:
        yoy_change_pct: Decimal | None
        if previous_total is None or previous_total <= 0:
            yoy_change_pct = None
        else:
            yoy_change_pct = ((total_spend - previous_total) / previous_total) * Decimal("100")
            yoy_change_pct = yoy_change_pct.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)
        if original_start_year is None or spend_year >= original_start_year:
            result.append((spend_year, total_spend, contract_count, yoy_change_pct))
        previous_total = total_spend
    return result


def _parse_bins(raw_bins: str | None) -> list[int] | None:
    if raw_bins is None or not raw_bins.strip():
        return None
    values: list[int] = []
    for part in raw_bins.split(","):
        normalized = "".join(ch for ch in part.strip() if ch.isdigit())
        if not normalized:
            continue
        values.append(int(normalized))
    if not values:
        return None
    return values


def _parse_code_list(raw_codes: str | None, *, code_length: int) -> list[str] | None:
    if raw_codes is None or not raw_codes.strip():
        return None
    values: list[str] = []
    for part in raw_codes.split(","):
        digits = "".join(ch for ch in part.strip() if ch.isdigit())
        if len(digits) >= code_length:
            values.append(digits[:code_length])
    if not values:
        return None
    return values


def _result_to_json_rows(
    result: Sequence[tuple[object, ...]], columns: Sequence[str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in result:
        row: dict[str, object] = {}
        for idx, column in enumerate(columns):
            value = item[idx] if idx < len(item) else None
            if isinstance(value, Decimal):
                row[column] = str(value)
            else:
                row[column] = value
        rows.append(row)
    return rows


def main(*, config_path: Path = DEFAULT_CONFIG_PATH) -> int:
    parser = argparse.ArgumentParser(
        description="Analytical queries for Kazakhstan procurement data"
    )
    _config_action = parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Path to config.yaml",
    )
    _query_action = parser.add_argument(
        "--query",
        choices=("total_spend", "enstr", "region", "concentration", "trends"),
        required=True,
        help="Query to execute",
    )
    _bins_action = parser.add_argument(
        "--bins",
        type=str,
        default=None,
        help="Comma-separated BIN filter",
    )
    _start_date_action = parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    _end_date_action = parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    _enstr_action = parser.add_argument(
        "--enstr-codes",
        type=str,
        default=None,
        help="Comma-separated ENSTR lvl4 codes",
    )
    _kato_action = parser.add_argument(
        "--kato-codes",
        type=str,
        default=None,
        help="Comma-separated KATO prefixes",
    )
    _parser_actions = (
        _config_action,
        _query_action,
        _bins_action,
        _start_date_action,
        _end_date_action,
        _enstr_action,
        _kato_action,
    )
    _ = _parser_actions

    args = parser.parse_args()
    parsed_config = cast(Path, args.config)
    parsed_query = cast(str, args.query)
    parsed_bins_raw = cast(str | None, args.bins)
    parsed_start_date_raw = cast(str | None, args.start_date)
    parsed_end_date_raw = cast(str | None, args.end_date)
    parsed_enstr_raw = cast(str | None, args.enstr_codes)
    parsed_kato_raw = cast(str | None, args.kato_codes)

    if parsed_start_date_raw is None and parsed_end_date_raw is None:
        LOGGER.warning(
            "Date range is required in CLI mode to avoid expensive unbounded analytics scans"
        )
        parser.error("Please provide --start-date and/or --end-date for bounded execution")

    start_date = _parse_datetime(parsed_start_date_raw) if parsed_start_date_raw else None
    end_date = _parse_datetime(parsed_end_date_raw) if parsed_end_date_raw else None

    bins = _parse_bins(parsed_bins_raw)
    enstr_codes = _parse_code_list(parsed_enstr_raw, code_length=4)
    kato_codes = _parse_code_list(parsed_kato_raw, code_length=2)

    config = _load_config(parsed_config)
    analytics_cfg = _as_object_mapping(config.get("analytics")) or {}
    min_sample_size = _to_uint64(
        analytics_cfg.get("min_sample_size"), default=DEFAULT_MIN_SAMPLE_SIZE
    )
    min_sample_size = max(DEFAULT_MIN_SAMPLE_SIZE, min_sample_size)

    clickhouse = _get_clickhouse_client()
    try:
        query_name = parsed_query
        payload: dict[str, object]
        if query_name == "total_spend":
            total_spend_result = total_spend_by_bin(
                clickhouse,
                bins=bins,
                start_date=start_date,
                end_date=end_date,
                enstr_codes=enstr_codes,
                kato_codes=kato_codes,
            )
            payload = {
                "query": query_name,
                "rows": _result_to_json_rows(
                    total_spend_result,
                    ["customer_bin", "total_contract_sum", "contract_count"],
                ),
            }
        elif query_name == "enstr":
            enstr_result = spend_by_enstr(
                clickhouse,
                bins=bins,
                start_date=start_date,
                end_date=end_date,
                enstr_codes=enstr_codes,
                kato_codes=kato_codes,
            )
            payload = {
                "query": query_name,
                "rows": _result_to_json_rows(
                    enstr_result,
                    ["enstr_code", "total_spend", "contract_count"],
                ),
            }
        elif query_name == "region":
            region_result = spend_by_region(
                clickhouse,
                bins=bins,
                start_date=start_date,
                end_date=end_date,
                enstr_codes=enstr_codes,
                kato_codes=kato_codes,
            )
            payload = {
                "query": query_name,
                "rows": _result_to_json_rows(
                    region_result,
                    ["kato_code", "region_name", "total_spend", "contract_count"],
                ),
            }
        elif query_name == "concentration":
            concentration_result = supplier_concentration(
                clickhouse,
                bins=bins,
                start_date=start_date,
                end_date=end_date,
                enstr_codes=enstr_codes,
                kato_codes=kato_codes,
                min_sample_size=min_sample_size,
            )
            payload = {
                "query": query_name,
                "rows": _result_to_json_rows(
                    concentration_result,
                    [
                        "customer_bin",
                        "hhi_score",
                        "supplier_count",
                        "top_supplier_share",
                    ],
                ),
                "hhi_interpretation": {
                    "competitive_lt": "0.15",
                    "moderate_range": "0.15-0.25",
                    "high_gt": "0.25",
                },
                "min_sample_size": min_sample_size,
            }
        else:
            trends_result = year_over_year_trends(
                clickhouse,
                bins=bins,
                start_date=start_date,
                end_date=end_date,
                enstr_codes=enstr_codes,
                kato_codes=kato_codes,
            )
            payload = {
                "query": query_name,
                "rows": _result_to_json_rows(
                    trends_result,
                    ["year", "total_spend", "contract_count", "yoy_change_pct"],
                ),
            }

        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        clickhouse.disconnect()


__all__ = [
    "ClickHouseClient",
    "total_spend_by_bin",
    "spend_by_enstr",
    "spend_by_region",
    "supplier_concentration",
    "year_over_year_trends",
    "main",
]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    raise SystemExit(main())
