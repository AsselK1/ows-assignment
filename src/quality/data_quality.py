from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Protocol, cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_OUTPUT_PATH = Path(".sisyphus/evidence/task-13-quality-report.txt")
TARGET_START = datetime(2024, 1, 1)
TARGET_END = datetime(2026, 12, 31, 23, 59, 59)
WARN_NULL_PCT = Decimal("1.00")
ERROR_NULL_PCT = Decimal("5.00")


class ClickHouseClient(Protocol):
    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]: ...

    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class RequiredField:
    name: str
    field_type: str


@dataclass
class FieldMetric:
    field: str
    invalid_count: int
    invalid_pct: Decimal


@dataclass
class TableResult:
    name: str
    total_rows: int
    required_field_metrics: list[FieldMetric] = field(default_factory=list)
    duplicate_key_count: int = 0
    duplicate_row_excess: int = 0
    referential_violations: dict[str, int] = field(default_factory=dict)
    date_range_violations: dict[str, int] = field(default_factory=dict)
    bin_scope_violations: int = 0
    status_violations: int = 0
    format_violations: dict[str, int] = field(default_factory=dict)
    type_violations: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    generated_at: datetime
    tables: list[TableResult]
    errors: list[str]
    warnings: list[str]
    quality_score: str


ENTITY_RULES: dict[str, dict[str, object]] = {
    "subjects": {
        "required_fields": [
            RequiredField("id", "numeric"),
            RequiredField("bin", "numeric"),
            RequiredField("name_ru", "string"),
        ],
        "duplicate_key": "id",
        "bin_scope_field": "bin",
        "type_expectations": {
            "id": ("UInt",),
            "bin": ("UInt",),
            "name_ru": ("String",),
        },
    },
    "plans": {
        "required_fields": [
            RequiredField("id", "numeric"),
            RequiredField("customer_bin", "numeric"),
            RequiredField("plan_year", "numeric"),
        ],
        "duplicate_key": "id",
        "date_fields": ["publish_date"],
        "year_fields": ["plan_year"],
        "bin_scope_field": "customer_bin",
        "type_expectations": {
            "id": ("UInt",),
            "customer_bin": ("UInt",),
            "plan_year": ("UInt",),
            "publish_date": ("DateTime",),
        },
    },
    "announcements": {
        "required_fields": [
            RequiredField("id", "numeric"),
            RequiredField("customer_bin", "numeric"),
            RequiredField("total_sum", "numeric"),
        ],
        "duplicate_key": "id",
        "date_fields": ["publish_date", "start_date", "end_date"],
        "bin_scope_field": "customer_bin",
        "status_field": "ref_buy_status_id",
        "type_expectations": {
            "id": ("UInt",),
            "customer_bin": ("UInt",),
            "total_sum": ("Decimal",),
            "publish_date": ("DateTime",),
            "ref_buy_status_id": ("UInt",),
        },
    },
    "lots": {
        "required_fields": [
            RequiredField("id", "numeric"),
            RequiredField("trd_buy_id", "numeric"),
            RequiredField("enstr_code", "string"),
        ],
        "duplicate_key": "id",
        "date_fields": ["created_at"],
        "bin_scope_field": "customer_bin",
        "referential_checks": {
            "trd_buy_id -> announcements.id": (
                "SELECT count() "
                "FROM lots l "
                "LEFT JOIN (SELECT DISTINCT id FROM announcements WHERE id > 0) a ON l.trd_buy_id = a.id "
                "WHERE l.trd_buy_id > 0 AND a.id IS NULL"
            ),
        },
        "format_checks": {
            "invalid_enstr_code": (
                "SELECT count() FROM lots "
                "WHERE enstr_code IS NULL "
                "OR trim(BOTH ' ' FROM enstr_code) = '' "
                "OR length(replaceRegexpAll(enstr_code, '[^0-9]', '')) < 2"
            ),
            "unknown_enstr_code": (
                "SELECT count() "
                "FROM lots l "
                "LEFT JOIN (SELECT DISTINCT enstr_code FROM reference_enstr) r ON l.enstr_code = r.enstr_code "
                "WHERE trim(BOTH ' ' FROM l.enstr_code) != '' AND r.enstr_code IS NULL"
            ),
        },
        "type_expectations": {
            "id": ("UInt",),
            "trd_buy_id": ("UInt",),
            "enstr_code": ("String",),
            "created_at": ("DateTime",),
        },
    },
    "contracts": {
        "required_fields": [
            RequiredField("id", "numeric"),
            RequiredField("customer_bin", "numeric"),
            RequiredField("supplier_bin", "numeric"),
            RequiredField("contract_sum", "numeric"),
        ],
        "duplicate_key": "id",
        "date_fields": ["sign_date", "start_date", "end_date"],
        "bin_scope_field": "customer_bin",
        "referential_checks": {
            "trd_buy_id -> announcements.id": (
                "SELECT count() "
                "FROM contracts c "
                "LEFT JOIN (SELECT DISTINCT id FROM announcements WHERE id > 0) a ON c.trd_buy_id = a.id "
                "WHERE c.trd_buy_id > 0 AND a.id IS NULL"
            ),
        },
        "type_expectations": {
            "id": ("UInt",),
            "customer_bin": ("UInt",),
            "supplier_bin": ("UInt",),
            "contract_sum": ("Decimal",),
            "sign_date": ("DateTime",),
        },
    },
    "contract_acts": {
        "required_fields": [
            RequiredField("id", "numeric"),
            RequiredField("contract_id", "numeric"),
            RequiredField("act_sum", "numeric"),
        ],
        "duplicate_key": "id",
        "date_fields": ["act_date", "approve_date"],
        "bin_scope_field": "customer_bin",
        "referential_checks": {
            "contract_id -> contracts.id": (
                "SELECT count() "
                "FROM contract_acts a "
                "LEFT JOIN (SELECT DISTINCT id FROM contracts WHERE id > 0) c ON a.contract_id = c.id "
                "WHERE a.contract_id > 0 AND c.id IS NULL"
            ),
        },
        "type_expectations": {
            "id": ("UInt",),
            "contract_id": ("UInt",),
            "act_sum": ("Decimal",),
            "act_date": ("DateTime",),
        },
    },
    "reference_enstr": {
        "required_fields": [
            RequiredField("enstr_code", "string"),
            RequiredField("name_ru", "string"),
            RequiredField("name_kz", "string"),
        ],
        "duplicate_key": "enstr_code",
        "type_expectations": {
            "enstr_code": ("String",),
            "name_ru": ("String",),
            "name_kz": ("String",),
        },
    },
    "reference_kato": {
        "required_fields": [
            RequiredField("kato_code", "string"),
            RequiredField("name_ru", "string"),
            RequiredField("name_kz", "string"),
        ],
        "duplicate_key": "kato_code",
        "type_expectations": {
            "kato_code": ("String",),
            "name_ru": ("String",),
            "name_kz": ("String",),
        },
    },
    "reference_mkei": {
        "required_fields": [
            RequiredField("mkei_code", "string"),
            RequiredField("name_ru", "string"),
            RequiredField("name_kz", "string"),
        ],
        "duplicate_key": "mkei_code",
        "type_expectations": {
            "mkei_code": ("String",),
            "name_ru": ("String",),
            "name_kz": ("String",),
        },
    },
}


def _now_utc_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _to_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def _to_pct(part: int, whole: int) -> Decimal:
    if whole <= 0:
        return Decimal("0.00")
    value = (Decimal(part) / Decimal(whole)) * Decimal("100")
    return value.quantize(Decimal("1.00"), rounding=ROUND_HALF_UP)


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


def _load_target_bins(config_path: Path) -> set[int]:
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

    bins: set[int] = set()
    for item in cast(list[object], bins_raw_obj):
        parsed = _to_uint64(item, default=0)
        if parsed > 0:
            bins.add(parsed)

    if len(bins) != 27:
        LOGGER.warning("Expected 27 BINs in config, got %d", len(bins))
    if not bins:
        raise ValueError("No valid BINs found in config.yaml")
    return bins


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
        username=user,
        password=password,
        database=database,
        secure=is_secure,
        connect_timeout=15,
        send_receive_timeout=30,
    )


def _scalar_count(clickhouse: ClickHouseClient, query: str, params: object | None = None) -> int:
    rows = clickhouse.execute(query, params)
    if not rows:
        return 0
    return _to_int(rows[0][0])


def _required_invalid_expr(field_name: str, field_type: str) -> str:
    if field_type == "string":
        return f"({field_name} IS NULL OR trim(BOTH ' ' FROM {field_name}) = '')"
    if field_type == "date":
        return f"({field_name} IS NULL)"
    return f"({field_name} IS NULL OR {field_name} = 0)"


def _check_required_fields(
    clickhouse: ClickHouseClient,
    table: str,
    total_rows: int,
    required_fields: Sequence[RequiredField],
) -> list[FieldMetric]:
    metrics: list[FieldMetric] = []
    for required in required_fields:
        expr = _required_invalid_expr(required.name, required.field_type)
        invalid_count = _scalar_count(
            clickhouse,
            f"SELECT countIf({expr}) FROM {table}",
        )
        metrics.append(
            FieldMetric(
                field=required.name,
                invalid_count=invalid_count,
                invalid_pct=_to_pct(invalid_count, total_rows),
            )
        )
    return metrics


def _check_duplicates(clickhouse: ClickHouseClient, table: str, key_field: str) -> tuple[int, int]:
    duplicate_keys = _scalar_count(
        clickhouse,
        f"SELECT count() FROM (SELECT {key_field}, count() AS c FROM {table} GROUP BY {key_field} HAVING c > 1)",
    )
    duplicate_excess = _scalar_count(
        clickhouse,
        (
            f"SELECT ifNull(sum(c - 1), 0) "
            f"FROM (SELECT count() AS c FROM {table} GROUP BY {key_field} HAVING c > 1)"
        ),
    )
    return duplicate_keys, duplicate_excess


def _check_date_range(
    clickhouse: ClickHouseClient,
    table: str,
    date_fields: Sequence[str],
    year_fields: Sequence[str],
) -> dict[str, int]:
    violations: dict[str, int] = {}
    for date_field in date_fields:
        count = _scalar_count(
            clickhouse,
            (
                f"SELECT countIf({date_field} < toDateTime(%(start)s) OR {date_field} > toDateTime(%(end)s)) "
                f"FROM {table}"
            ),
            {
                "start": TARGET_START.strftime("%Y-%m-%d %H:%M:%S"),
                "end": TARGET_END.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        violations[date_field] = count

    for year_field in year_fields:
        count = _scalar_count(
            clickhouse,
            f"SELECT countIf({year_field} < 2024 OR {year_field} > 2026) FROM {table}",
        )
        violations[year_field] = count

    return violations


def _check_bin_scope(
    clickhouse: ClickHouseClient,
    table: str,
    bin_field: str,
    target_bins: set[int],
) -> int:
    if not target_bins:
        return 0
    return _scalar_count(
        clickhouse,
        f"SELECT countIf({bin_field} > 0 AND {bin_field} NOT IN %(bins)s) FROM {table}",
        {"bins": tuple(sorted(target_bins))},
    )


def _check_data_types(
    clickhouse: ClickHouseClient,
    table: str,
    type_expectations: Mapping[str, tuple[str, ...]],
) -> list[str]:
    violations: list[str] = []
    if not type_expectations:
        return violations

    rows = clickhouse.execute(
        (
            "SELECT name, type FROM system.columns "
            "WHERE database = currentDatabase() AND table = %(table)s AND name IN %(columns)s"
        ),
        {"table": table, "columns": tuple(type_expectations.keys())},
    )
    actual: dict[str, str] = {
        str(row[0]): str(row[1])
        for row in rows
        if len(row) >= 2 and row[0] is not None and row[1] is not None
    }

    for column, expected_prefixes in type_expectations.items():
        actual_type = actual.get(column)
        if actual_type is None:
            violations.append(f"{table}.{column} is missing in schema")
            continue
        if not any(actual_type.startswith(prefix) for prefix in expected_prefixes):
            expected_text = ", ".join(expected_prefixes)
            violations.append(
                f"{table}.{column} type mismatch: expected {expected_text}, actual {actual_type}"
            )
    return violations


def _check_status_validity(clickhouse: ClickHouseClient, table: str, status_field: str) -> int:
    return _scalar_count(clickhouse, f"SELECT countIf({status_field} = 0) FROM {table}")


def _run_table_checks(
    clickhouse: ClickHouseClient,
    table: str,
    rule: Mapping[str, object],
    target_bins: set[int],
) -> TableResult:
    total_rows = _scalar_count(clickhouse, f"SELECT count() FROM {table}")
    required_fields = cast(list[RequiredField], rule.get("required_fields", []))
    duplicate_key = cast(str | None, rule.get("duplicate_key"))

    result = TableResult(name=table, total_rows=total_rows)
    result.required_field_metrics = _check_required_fields(
        clickhouse, table, total_rows, required_fields
    )

    if duplicate_key:
        duplicate_key_count, duplicate_row_excess = _check_duplicates(
            clickhouse, table, duplicate_key
        )
        result.duplicate_key_count = duplicate_key_count
        result.duplicate_row_excess = duplicate_row_excess

    referential_checks = cast(dict[str, str], rule.get("referential_checks", {}))
    for label, query in referential_checks.items():
        result.referential_violations[label] = _scalar_count(clickhouse, query)

    date_fields = cast(list[str], rule.get("date_fields", []))
    year_fields = cast(list[str], rule.get("year_fields", []))
    result.date_range_violations = _check_date_range(clickhouse, table, date_fields, year_fields)

    bin_scope_field = cast(str | None, rule.get("bin_scope_field"))
    if bin_scope_field:
        result.bin_scope_violations = _check_bin_scope(
            clickhouse, table, bin_scope_field, target_bins
        )

    status_field = cast(str | None, rule.get("status_field"))
    if status_field:
        result.status_violations = _check_status_validity(clickhouse, table, status_field)

    format_checks = cast(dict[str, str], rule.get("format_checks", {}))
    for label, query in format_checks.items():
        result.format_violations[label] = _scalar_count(clickhouse, query)

    type_expectations = cast(dict[str, tuple[str, ...]], rule.get("type_expectations", {}))
    result.type_violations = _check_data_types(clickhouse, table, type_expectations)
    return result


def _summarize_alerts(table_results: Sequence[TableResult]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for table in table_results:
        for field_metric in table.required_field_metrics:
            if field_metric.invalid_pct > ERROR_NULL_PCT:
                errors.append(
                    (
                        f"{table.name}.{field_metric.field}: {field_metric.invalid_pct}% invalid "
                        f"(>{ERROR_NULL_PCT}%)"
                    )
                )
            elif field_metric.invalid_pct >= WARN_NULL_PCT:
                warnings.append(
                    (
                        f"{table.name}.{field_metric.field}: {field_metric.invalid_pct}% invalid "
                        f"(between {WARN_NULL_PCT}% and {ERROR_NULL_PCT}%)"
                    )
                )

        if table.duplicate_row_excess > 0:
            warnings.append(
                (
                    f"{table.name}: duplicate keys={table.duplicate_key_count}, "
                    f"excess rows={table.duplicate_row_excess}"
                )
            )

        for relation, violation_count in table.referential_violations.items():
            if violation_count > 0:
                errors.append(f"{table.name} referential integrity {relation}: {violation_count}")

        for date_field, violation_count in table.date_range_violations.items():
            if violation_count > 0:
                warnings.append(f"{table.name}.{date_field} out-of-range rows: {violation_count}")

        if table.bin_scope_violations > 0:
            warnings.append(f"{table.name} BIN scope violations: {table.bin_scope_violations}")

        if table.status_violations > 0:
            warnings.append(f"{table.name} status validity violations: {table.status_violations}")

        for label, count in table.format_violations.items():
            if count > 0:
                warnings.append(f"{table.name} {label}: {count}")

        for type_violation in table.type_violations:
            errors.append(type_violation)

    return errors, warnings


def _quality_score(errors: Sequence[str], warnings: Sequence[str]) -> str:
    if errors:
        return "FAIL"
    if warnings:
        return "WARN"
    return "PASS"


def _format_table_block(table: TableResult) -> list[str]:
    lines: list[str] = [
        f"Table: {table.name}",
        f"  Total rows: {table.total_rows}",
        "  Required fields:",
    ]
    for metric in table.required_field_metrics:
        lines.append(
            f"    - {metric.field}: invalid={metric.invalid_count} ({metric.invalid_pct}%)"
        )

    lines.append(
        f"  Duplicates: keys={table.duplicate_key_count}, excess_rows={table.duplicate_row_excess}"
    )

    if table.referential_violations:
        lines.append("  Referential integrity:")
        for label, count in table.referential_violations.items():
            lines.append(f"    - {label}: {count}")

    if table.date_range_violations:
        lines.append("  Date range violations:")
        for field_name, count in table.date_range_violations.items():
            lines.append(f"    - {field_name}: {count}")

    lines.append(f"  BIN scope violations: {table.bin_scope_violations}")

    if table.status_violations:
        lines.append(f"  Status validity violations: {table.status_violations}")

    if table.format_violations:
        lines.append("  Format checks:")
        for label, count in table.format_violations.items():
            lines.append(f"    - {label}: {count}")

    if table.type_violations:
        lines.append("  Data type issues:")
        for issue in table.type_violations:
            lines.append(f"    - {issue}")

    return lines


def _build_report_text(report: QualityReport) -> str:
    lines: list[str] = [
        "Kazakhstan Procurement Data Quality Report",
        f"Generated at: {report.generated_at.isoformat()}Z",
        f"Target date range: {TARGET_START.date()} to {TARGET_END.date()}",
        f"Overall quality score: {report.quality_score}",
        "",
        "=== Table Metrics ===",
    ]

    for table in report.tables:
        lines.extend(_format_table_block(table))
        lines.append("")

    lines.append("=== Alerts ===")
    if not report.errors and not report.warnings:
        lines.append("No data quality alerts detected.")
    else:
        if report.errors:
            lines.append("ERROR:")
            for error in report.errors:
                lines.append(f"  - {error}")
        if report.warnings:
            lines.append("WARN:")
            for warning in report.warnings:
                lines.append(f"  - {warning}")

    lines.append("")
    lines.append("Policy:")
    lines.append("  - ERROR when required-field invalid ratio > 5%")
    lines.append("  - WARN when required-field invalid ratio is 1%-5%")
    lines.append("  - ERROR when referential integrity violations exist")
    lines.append("  - WARN when date range violations exist")
    return "\n".join(lines) + "\n"


def run_data_quality_validation(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> QualityReport:
    target_bins = _load_target_bins(config_path)
    clickhouse = _get_clickhouse_client()
    try:
        table_results: list[TableResult] = []
        for table_name, rule in ENTITY_RULES.items():
            table_result = _run_table_checks(clickhouse, table_name, rule, target_bins)
            table_results.append(table_result)
            LOGGER.info(
                "Quality metrics table=%s rows=%d duplicates=%d bin_scope=%d",
                table_result.name,
                table_result.total_rows,
                table_result.duplicate_row_excess,
                table_result.bin_scope_violations,
            )

        errors, warnings = _summarize_alerts(table_results)
        quality_score = _quality_score(errors, warnings)
        report = QualityReport(
            generated_at=_now_utc_naive(),
            tables=table_results,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
        )

        report_text = _build_report_text(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _ = output_path.write_text(report_text, encoding="utf-8")
        LOGGER.info(
            "Data quality report written to %s (score=%s, errors=%d, warnings=%d)",
            output_path,
            quality_score,
            len(errors),
            len(warnings),
        )
        return report
    finally:
        clickhouse.disconnect()


def _write_failure_report(output_path: Path, error: Exception) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_text = (
        "Kazakhstan Procurement Data Quality Report\n"
        f"Generated at: {_now_utc_naive().isoformat()}Z\n"
        "Overall quality score: FAIL\n\n"
        "Validation could not be executed in this environment.\n"
        f"Reason: {error}\n\n"
        "Runtime requirements:\n"
        "  - CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_DB, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD\n"
        "  - Reachable ClickHouse instance with all 9 tables populated\n"
    )
    _ = output_path.write_text(failure_text, encoding="utf-8")


def main(config_path: Path = DEFAULT_CONFIG_PATH, output_path: Path = DEFAULT_OUTPUT_PATH) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        report = run_data_quality_validation(config_path=config_path, output_path=output_path)
    except Exception as error:  # noqa: BLE001
        LOGGER.error("Data quality validation failed: %s", error)
        _write_failure_report(output_path, error)
        return 1

    if report.errors:
        LOGGER.error("Data quality validation completed with ERROR-level findings")
        return 2
    if report.warnings:
        LOGGER.warning("Data quality validation completed with WARN-level findings")
        return 0
    LOGGER.info("Data quality validation passed with no findings")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data quality validation on loaded entities")
    _config_action = parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    _output_action = parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write data quality report",
    )
    _ = (_config_action, _output_action)

    args = parser.parse_args()
    exit_code = main(config_path=cast(Path, args.config), output_path=cast(Path, args.output))
    raise SystemExit(exit_code)
