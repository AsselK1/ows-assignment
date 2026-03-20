from __future__ import annotations

import argparse
import calendar
import importlib
import json
import logging
import os
import re
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, NamedTuple, Protocol, TypedDict, cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, ValidationError

from src.agent.classifier import ClassificationResult, Entities, Language, classify
from src.analytics.queries import (
    spend_by_enstr,
    spend_by_region,
    supplier_concentration,
    total_spend_by_bin,
    year_over_year_trends,
)
from src.llm.client import LLMClient
from src.rag.retriever import retrieve

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH: Final[Path] = Path("config.yaml")
DEFAULT_PORTAL_BASE_URL: Final[str] = "https://ows.goszakup.gov.kz/v3"
DEFAULT_RETRIEVE_LIMIT: Final[int] = 10
DEFAULT_RETRIEVE_MAX_TOKENS: Final[int] = 4000
TOP_EXAMPLES_K: Final[int] = 5

# Query-type aware retrieval limits (for LLM context)
RETRIEVE_LIMITS_BY_QUERY_TYPE: Final[dict[str, int]] = {
    "SEARCH": 10,
    "COMPARISON": 50,
    "ANALYTICS": 30,
    "ANOMALY_DETECTION": 100,
    "FAIRNESS": 50,
}

SupportedQueryType = Literal[
    "SEARCH",
    "COMPARISON",
    "ANALYTICS",
    "ANOMALY_DETECTION",
    "FAIRNESS",
]


class RetrievedChunk(TypedDict):
    content: str
    entity_type: str
    procurement_id: int
    score: float
    metadata: dict[str, object]


class DataUsedSection(TypedDict):
    period: str
    filters: dict[str, list[str]]
    entities: dict[str, int]
    sample_size: int


class ComparisonSection(TypedDict):
    weighted_average: float | None
    median: float | None
    outliers: list[dict[str, object]]
    reference_baseline: str
    methodology: str


class EvaluationMetricSection(TypedDict):
    metric_name: str
    formula: str
    threshold: str
    value: float | None


class LimitationsConfidenceSection(TypedDict):
    limitations: list[str]
    data_completeness: float
    temporal_coverage: str
    confidence: float


class ExampleItem(TypedDict):
    procurement_id: int
    entity_type: str
    contract_sum: float | None
    supplier_name: str | None
    sign_date: str | None
    anomaly_score: float | None


class ResponseSections(TypedDict):
    summary: str
    data_used: DataUsedSection
    comparison: ComparisonSection
    evaluation_metric: EvaluationMetricSection
    limitations_and_confidence: LimitationsConfidenceSection
    examples: list[ExampleItem]
    links: list[str]


class ExplainabilityMetadata(TypedDict):
    sample_size: int
    methodology: str
    confidence: float
    comparison_methodology: str
    query_type: SupportedQueryType
    language: Language


class GeneratedResponse(TypedDict):
    query: str
    query_type: SupportedQueryType
    language: Language
    answer: str
    sections: ResponseSections
    metadata: ExplainabilityMetadata


class _LLMSectionsModel(BaseModel):
    summary: str
    comparison: str
    evaluation_metric: str
    limitations_and_confidence: str


class ClickHouseClient(Protocol):
    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]: ...

    def disconnect(self) -> None: ...


class _StdoutBuffer(Protocol):
    def write(self, data: bytes) -> object: ...

    def flush(self) -> None: ...


def _module_attr(module: ModuleType, name: str) -> object:
    return cast(object, getattr(module, name))


def _build_instance(factory: object, *args: object, **kwargs: object) -> object:
    callable_factory = cast(Callable[..., object], factory)
    return callable_factory(*args, **kwargs)


def _load_yaml(path: Path) -> object:
    yaml_module = importlib.import_module("yaml")
    safe_load = cast(Callable[[str], object], _module_attr(yaml_module, "safe_load"))
    return safe_load(path.read_text(encoding="utf-8"))


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return None


def _load_config(config_path: Path) -> Mapping[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_obj = _load_yaml(config_path)
    config = _as_mapping(config_obj)
    if config is None:
        raise ValueError("config.yaml must be a dictionary at top level")
    return config


def _to_text(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _to_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return default
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return default
    try:
        return int(digits)
    except ValueError:
        return default


def _to_float(value: object, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "")
    if not text:
        return default
    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return default


_DATE_GRANULARITY = Literal["day", "month", "year"]

_MONTH_TOKENS: Final[dict[str, int]] = {
    # Russian genitive
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
    # Russian prepositional ("в январе")
    "январе": 1,
    "феврале": 2,
    "марте": 3,
    "апреле": 4,
    "мае": 5,
    "июне": 6,
    "июле": 7,
    "августе": 8,
    "сентябре": 9,
    "октябре": 10,
    "ноябре": 11,
    "декабре": 12,
    # Russian accusative ("за январь")
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
    # Kazakh
    "қаңтар": 1,
    "ақпан": 2,
    "наурыз": 3,
    "сәуір": 4,
    "мамыр": 5,
    "маусым": 6,
    "шілде": 7,
    "тамыз": 8,
    "қыркүйек": 9,
    "қазан": 10,
    "қараша": 11,
    "желтоқсан": 12,
}


class _ParsedDate(NamedTuple):
    dt: datetime
    granularity: _DATE_GRANULARITY


def _parse_date(value: str) -> datetime | None:
    result = _parse_date_with_granularity(value)
    return result.dt if result is not None else None


def _parse_date_with_granularity(value: str) -> _ParsedDate | None:
    normalized = value.strip()
    if not normalized:
        return None

    if re.fullmatch(r"\d{4}", normalized):
        try:
            return _ParsedDate(datetime(int(normalized), 1, 1), "year")
        except ValueError:
            return None

    for pattern in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return _ParsedDate(datetime.strptime(normalized, pattern), "day")
        except ValueError:
            continue

    # "январе 2025" / "января 2025" / "қаңтар 2025" etc.
    match = re.fullmatch(r"([\wәғқңөұүһіё]+)\s+(\d{4})", normalized.lower())
    if match:
        month_value = _MONTH_TOKENS.get(match.group(1))
        year_value = _to_int(match.group(2), default=0)
        if month_value is not None and year_value > 0:
            try:
                return _ParsedDate(datetime(year_value, month_value, 1), "month")
            except ValueError:
                return None
    return None


def _period_end_for_granularity(dt: datetime, granularity: _DATE_GRANULARITY) -> datetime:
    if granularity == "day":
        return dt.replace(hour=23, minute=59, second=59)
    if granularity == "month":
        last_day = calendar.monthrange(dt.year, dt.month)[1]
        return dt.replace(day=last_day, hour=23, minute=59, second=59)
    return dt.replace(month=12, day=31, hour=23, minute=59, second=59)


def _resolve_individual_periods(
    entities: Entities, config: Mapping[str, object]
) -> list[tuple[datetime, datetime, str]]:
    parsed = [
        result
        for result in (_parse_date_with_granularity(token) for token in entities.get("dates", []))
        if result is not None
    ]
    if len(parsed) < 2:
        return [_resolve_period(entities, config)]
    periods: list[tuple[datetime, datetime, str]] = []
    for entry in sorted(parsed, key=lambda e: e.dt):
        start = entry.dt
        end = _period_end_for_granularity(entry.dt, entry.granularity)
        label = f"{start.date().isoformat()} - {end.date().isoformat()}"
        periods.append((start, end, label))
    return periods


def _resolve_period(
    entities: Entities, config: Mapping[str, object]
) -> tuple[datetime, datetime, str]:
    parsed = [
        result
        for result in (_parse_date_with_granularity(token) for token in entities.get("dates", []))
        if result is not None
    ]

    data_cfg = _as_mapping(config.get("data"))
    range_cfg = _as_mapping(data_cfg.get("date_range") if data_cfg is not None else None)
    default_start = _parse_date(
        _to_text(range_cfg.get("start") if range_cfg is not None else "2024-01-01")
    )
    default_end = _parse_date(
        _to_text(range_cfg.get("end") if range_cfg is not None else "2026-12-31")
    )

    today = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)

    if not parsed:
        start = default_start or datetime(2024, 1, 1)
        end = min(default_end or today, today)
    elif len(parsed) == 1:
        # Single month/year token: expand end to cover full granularity window
        # e.g. "января 2025" → 2025-01-01 00:00:00 to 2025-01-31 23:59:59
        entry = parsed[0]
        start = entry.dt
        end = _period_end_for_granularity(entry.dt, entry.granularity)
    else:
        start = min(entry.dt for entry in parsed)
        latest = max(parsed, key=lambda e: e.dt)
        end = _period_end_for_granularity(latest.dt, latest.granularity)

    if end < start:
        start, end = end, start

    return start, end, f"{start.date().isoformat()} - {end.date().isoformat()}"


def _build_retrieval_filters(
    entities: Entities, start: datetime, end: datetime
) -> dict[str, object]:
    filters: dict[str, object] = {}
    bins = [_to_int(item, default=0) for item in entities.get("bins", [])]
    bins = [value for value in bins if value > 0]
    if bins:
        filters["customer_bin"] = bins
    enstr_codes = entities.get("enstr_codes", [])
    if enstr_codes:
        filters["enstr_code"] = enstr_codes[0]
    filters["date_from"] = start.isoformat(timespec="seconds")
    filters["date_to"] = end.isoformat(timespec="seconds")
    return filters


def _normalize_chunk(item: Mapping[str, object]) -> RetrievedChunk | None:
    content = _to_text(item.get("content"))
    procurement_id = _to_int(item.get("procurement_id"), default=0)
    if not content or procurement_id <= 0:
        return None

    metadata_obj = item.get("metadata")
    metadata: dict[str, object] = {}
    if isinstance(metadata_obj, Mapping):
        typed_metadata = cast(Mapping[str, object], metadata_obj)
        for key, value in typed_metadata.items():
            metadata[str(key)] = value
    return {
        "content": content,
        "entity_type": _to_text(item.get("entity_type"), default="unknown"),
        "procurement_id": procurement_id,
        "score": _to_float(item.get("score"), default=0.0),
        "metadata": metadata,
    }


def _retrieve_context(
    query: str,
    filters: dict[str, object],
    query_type: SupportedQueryType = "SEARCH",
) -> list[RetrievedChunk]:
    limit = RETRIEVE_LIMITS_BY_QUERY_TYPE.get(query_type, DEFAULT_RETRIEVE_LIMIT)
    try:
        raw_chunks = retrieve(
            query,
            filters=filters,
            limit=limit,
            max_tokens=DEFAULT_RETRIEVE_MAX_TOKENS,
        )
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("retrieval_failed error=%s", type(error).__name__)
        return []
    chunks: list[RetrievedChunk] = []
    for item in raw_chunks:
        normalized = _normalize_chunk(item)
        if normalized is not None:
            chunks.append(normalized)
    return chunks


def _get_clickhouse_client() -> ClickHouseClient:
    host = os.getenv("CLICKHOUSE_HOST")
    if not host:
        raise ValueError("CLICKHOUSE_HOST is required")
    port = _to_int(os.getenv("CLICKHOUSE_PORT", "9440"), default=9440)
    database = _to_text(os.getenv("CLICKHOUSE_DB"))
    if not database:
        raise ValueError("CLICKHOUSE_DB is required")
    user = _to_text(os.getenv("CLICKHOUSE_USER"), default="default")
    password = _to_text(os.getenv("CLICKHOUSE_PASSWORD"), default="")

    is_secure = port in (9440, 8443)
    clickhouse_driver = importlib.import_module("clickhouse_driver")
    client_factory = _module_attr(clickhouse_driver, "Client")
    return cast(
        ClickHouseClient,
        _build_instance(
            client_factory,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            secure=is_secure,
            connect_timeout=15,
            send_receive_timeout=30,
        ),
    )


def _safe_analytics(
    bins: list[int],
    enstr_codes: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "total_spend_by_bin": [],
        "spend_by_enstr": [],
        "spend_by_region": [],
        "supplier_concentration": [],
        "year_over_year_trends": [],
        "contracts_analyzed": 0,
        "lots_analyzed": 0,
        "plans_analyzed": 0,
    }
    try:
        clickhouse_client = _get_clickhouse_client()
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("analytics_client_unavailable error=%s", type(error).__name__)
        return payload

    try:
        bins_param = bins or None
        enstr_param = enstr_codes or None

        total_spend_result = total_spend_by_bin(
            clickhouse_client,
            bins=bins_param,
            start_date=start,
            end_date=end,
            enstr_codes=enstr_param,
        )
        payload["total_spend_by_bin"] = total_spend_result
        contracts_count = sum(row[2] for row in total_spend_result)
        payload["contracts_analyzed"] = contracts_count

        payload["spend_by_enstr"] = spend_by_enstr(
            clickhouse_client,
            bins=bins_param,
            start_date=start,
            end_date=end,
            enstr_codes=enstr_param,
        )
        payload["spend_by_region"] = spend_by_region(
            clickhouse_client,
            bins=bins_param,
            start_date=start,
            end_date=end,
            enstr_codes=enstr_param,
        )
        payload["supplier_concentration"] = supplier_concentration(
            clickhouse_client,
            bins=bins_param,
            start_date=start,
            end_date=end,
            enstr_codes=enstr_param,
        )
        payload["year_over_year_trends"] = year_over_year_trends(
            clickhouse_client,
            bins=bins_param,
            start_date=start,
            end_date=end,
            enstr_codes=enstr_param,
        )
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("analytics_query_failed error=%s", type(error).__name__)
    finally:
        clickhouse_client.disconnect()

    return payload


def _calculate_data_completeness(chunks: Sequence[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    complete = 0
    for chunk in chunks:
        sign_date = _to_text(chunk["metadata"].get("sign_date"))
        if chunk["procurement_id"] > 0 and sign_date:
            complete += 1
    return max(0.0, min(1.0, complete / len(chunks)))


def _calculate_confidence(
    actual_sample_size: int,
    entities: Entities,
    chunks: Sequence[RetrievedChunk] | None = None,
    query_type: SupportedQueryType = "SEARCH",
    data_completeness: float = 1.0,
    quality_score: str = "PASS",
) -> float:
    """Calculate confidence score based on multiple signals.

    This implements the Terms of Reference requirement for confidence to reflect
    "data quality assessment and risks".

    Args:
        actual_sample_size: Number of contracts/records analyzed
        entities: Extracted entities from classification
        chunks: Retrieved chunks for computing average retrieval score
        query_type: Type of query for reliability factor
        data_completeness: Data completeness ratio (0.0-1.0)
        quality_score: Data quality from quality module (PASS/WARN/FAIL)

    Returns:
        Confidence score from 0.0 to 1.0
    """
    if actual_sample_size == 0:
        return 0.0

    # Sample size factor
    if actual_sample_size < 5:
        n_factor = 0.20
    elif actual_sample_size < 10:
        n_factor = 0.35
    elif actual_sample_size < 20:
        n_factor = 0.50
    elif actual_sample_size < 50:
        n_factor = 0.60
    elif actual_sample_size < 100:
        n_factor = 0.65
    else:
        n_factor = 0.70

    # Retrieval quality factor from chunks
    avg_retrieval_score = 0.5
    if chunks:
        scores = [c["score"] for c in chunks if c.get("score", 0) > 0]
        if scores:
            avg_retrieval_score = sum(scores) / len(scores)
    retrieval_factor = avg_retrieval_score * 0.08

    # Data completeness factor
    completeness_factor = data_completeness * 0.08

    # Query type reliability factor
    query_type_factors = {
        "SEARCH": 0.10,
        "COMPARISON": 0.08,
        "ANALYTICS": 0.08,
        "ANOMALY_DETECTION": 0.05,
        "FAIRNESS": 0.05,
    }
    query_factor = query_type_factors.get(query_type, 0.05)

    # Data quality penalty
    quality_penalties = {"PASS": 0.0, "WARN": -0.10, "FAIL": -0.25}
    quality_penalty = quality_penalties.get(quality_score, 0.0)

    # Entity extraction boost
    entity_boost = 0.03 if any(entities.values()) else 0.0

    confidence = (
        n_factor
        + retrieval_factor
        + completeness_factor
        + query_factor
        + quality_penalty
        + entity_boost
    )

    return round(max(0.0, min(1.0, confidence)), 3)


def _numbers_for_comparison(
    chunks: Sequence[RetrievedChunk], analytics_payload: Mapping[str, object]
) -> list[float]:
    values: list[float] = []
    enstr_rows_obj = analytics_payload.get("spend_by_enstr", [])
    if isinstance(enstr_rows_obj, Sequence):
        for row in enstr_rows_obj:
            if isinstance(row, Sequence) and len(row) >= 3:
                total = _to_float(row[1], default=0.0)
                count = _to_float(row[2], default=0.0)
                if total > 0 and count > 0:
                    values.append(total / count)
    if values:
        return values
    return [chunk["score"] for chunk in chunks if chunk["score"] > 0]


def _build_comparison_section(
    chunks: Sequence[RetrievedChunk],
    analytics_payload: Mapping[str, object],
    *,
    query_type: SupportedQueryType,
) -> ComparisonSection:
    values = _numbers_for_comparison(chunks, analytics_payload)
    if not values:
        return {
            "weighted_average": None,
            "median": None,
            "outliers": [],
            "reference_baseline": "No comparable numeric baseline was available.",
            "methodology": "No statistical comparison due to missing numeric sample.",
        }

    weighted_average = sum(values) / len(values)
    median_value = statistics.median(values)
    std_dev = statistics.pstdev(values) if len(values) > 1 else 0.0

    outliers: list[dict[str, object]] = []
    if std_dev > 0:
        for chunk in chunks:
            score = chunk["score"]
            if score <= 0:
                continue
            z = abs((score - weighted_average) / std_dev)
            if z > 2.0:
                outliers.append(
                    {
                        "procurement_id": chunk["procurement_id"],
                        "entity_type": chunk["entity_type"],
                        "value": round(score, 4),
                        "z_score": round(z, 3),
                    }
                )

    baseline_text = (
        "Market baseline by ENSTR weighted group averages"
        if query_type in {"COMPARISON", "FAIRNESS", "ANOMALY_DETECTION"}
        else "Retrieved sample baseline"
    )
    return {
        "weighted_average": round(weighted_average, 4),
        "median": round(float(median_value), 4),
        "outliers": outliers,
        "reference_baseline": baseline_text,
        "methodology": "weighted average + median + outliers beyond 2σ",
    }


def _metric_for_query_type(query_type: SupportedQueryType) -> EvaluationMetricSection:
    if query_type == "SEARCH":
        return {
            "metric_name": "Relevance score",
            "formula": "hybrid_score = 0.6 * rerank + 0.4 * fused(vector,bm25)",
            "threshold": ">= 0.50 considered relevant",
            "value": None,
        }
    if query_type == "COMPARISON":
        return {
            "metric_name": "Relative deviation",
            "formula": "deviation_pct = |value - weighted_avg| / weighted_avg * 100",
            "threshold": "> 30% is material deviation",
            "value": None,
        }
    if query_type == "ANALYTICS":
        return {
            "metric_name": "Year-over-year change",
            "formula": "yoy_pct = (current - previous) / previous * 100",
            "threshold": "informational trend metric",
            "value": None,
        }
    if query_type == "ANOMALY_DETECTION":
        return {
            "metric_name": "Anomaly deviation score",
            "formula": "z_score = |value - mean| / std; anomaly if z_score > 2",
            "threshold": "z_score > 2.0 or deviation > 30%",
            "value": None,
        }
    return {
        "metric_name": "Fairness score",
        "formula": "fairness = contract_sum / median(similar contracts)",
        "threshold": "outside 0.7-1.3 indicates potential unfairness",
        "value": None,
    }


def _extract_contract_sum(content: str, metadata: Mapping[str, object]) -> float | None:
    direct = _to_float(metadata.get("contract_sum"), default=0.0)
    if direct > 0:
        return round(direct, 2)
    match = re.search(
        r"(?:amount|sum|сумма)\s+([0-9][0-9\s\.,]*[0-9])", content, flags=re.IGNORECASE
    )
    if not match:
        return None
    parsed = _to_float(match.group(1), default=0.0)
    return round(parsed, 2) if parsed > 0 else None


def _build_examples(chunks: Sequence[RetrievedChunk]) -> list[ExampleItem]:
    examples: list[ExampleItem] = []
    for chunk in chunks[:TOP_EXAMPLES_K]:
        metadata = chunk["metadata"]
        supplier_name = _to_text(metadata.get("supplier_name")) or None
        if supplier_name is None:
            raw_bin = metadata.get("supplier_bin")
            if raw_bin is not None:
                try:
                    bin_val = int(str(raw_bin))
                except (TypeError, ValueError):
                    bin_val = 0
                if bin_val > 0:
                    supplier_name = f"BIN {bin_val}"
        examples.append(
            {
                "procurement_id": chunk["procurement_id"],
                "entity_type": chunk["entity_type"],
                "contract_sum": _extract_contract_sum(chunk["content"], metadata),
                "supplier_name": supplier_name,
                "sign_date": _to_text(metadata.get("sign_date")) or None,
                "anomaly_score": round(chunk["score"], 4) if chunk["score"] > 0 else None,
            }
        )
    return examples


def _portal_base_url(config: Mapping[str, object]) -> str:
    from_env = _to_text(os.getenv("OWS_PORTAL_BASE_URL"))
    if from_env:
        return from_env.rstrip("/")

    ows_cfg = _as_mapping(config.get("ows"))
    if ows_cfg is not None:
        candidate = _to_text(ows_cfg.get("portal_base_url"))
        if candidate:
            return candidate.rstrip("/")

    api_cfg = _as_mapping(config.get("api"))
    if api_cfg is not None:
        candidate = _to_text(api_cfg.get("base_url"))
        if candidate:
            return candidate.rstrip("/")

    return DEFAULT_PORTAL_BASE_URL


def _entity_path(entity_type: str) -> str:
    normalized = entity_type.lower()
    if normalized == "contract":
        return "contract"
    if normalized == "lot":
        return "lots"
    if normalized == "announcement":
        return "trd-buy"
    if normalized == "plan":
        return "plans"
    return normalized


def _generate_links(chunks: Sequence[RetrievedChunk], config: Mapping[str, object]) -> list[str]:
    base = _portal_base_url(config)
    links: list[str] = []
    seen: set[str] = set()
    for chunk in chunks[:TOP_EXAMPLES_K]:
        path = _entity_path(chunk["entity_type"])
        url = f"{base}/{path}/{chunk['procurement_id']}"
        if url in seen:
            continue
        seen.add(url)
        links.append(url)
    return links


def _default_summary(query_type: SupportedQueryType, sample_size: int) -> str:
    if sample_size == 0:
        return (
            "No procurement records matched the current filters; no factual conclusion can be made."
        )
    if query_type == "SEARCH":
        return f"Retrieved {sample_size} relevant procurement records for the request."
    if query_type == "COMPARISON":
        return (
            f"Compared {sample_size} records and computed baseline statistics from available data."
        )
    if query_type == "ANALYTICS":
        return f"Analyzed {sample_size} records and prepared aggregate indicators for the selected period."
    if query_type == "ANOMALY_DETECTION":
        return f"Analyzed {sample_size} records for deviation patterns; outlier indicators are reported below."
    return f"Evaluated fairness context across {sample_size} comparable procurement records."


def _llm_language_name(language: Language) -> str:
    if language == "kazakh":
        return "Kazakh"
    return "Russian"


def _build_llm_messages(
    query: str,
    language: Language,
    query_type: SupportedQueryType,
    data_used: DataUsedSection,
    comparison: ComparisonSection,
    metric: EvaluationMetricSection,
    limitations: LimitationsConfidenceSection,
    examples: Sequence[ExampleItem],
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a procurement analyst. Use ONLY provided factual data. "
        "No speculation. Return JSON object with keys: "
        "summary, comparison, evaluation_metric, limitations_and_confidence."
    )
    payload = {
        "query": query,
        "query_type": query_type,
        "data_used": data_used,
        "comparison": comparison,
        "evaluation_metric": metric,
        "limitations_and_confidence": limitations,
        "examples": examples,
    }
    user_prompt = (
        f"Language: {_llm_language_name(language)}\n"
        "Rewrite concise factual text for each field using only input JSON.\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_answer_llm_messages(
    query: str,
    language: Language,
    query_type: SupportedQueryType,
    data_used: DataUsedSection,
    comparison: ComparisonSection,
    metric: EvaluationMetricSection,
    limitations: LimitationsConfidenceSection,
    examples: Sequence[ExampleItem],
    links: Sequence[str],
) -> list[dict[str, str]]:
    lang_name = _llm_language_name(language)
    system_prompt = (
        "You are a helpful procurement analyst assistant for Kazakhstan's public procurement system. "
        "You answer user questions in a clear, natural, human-readable way. "
        f"Always respond in {lang_name}. "
        "Use ONLY the factual data provided below — do not speculate or invent numbers. "
        "Structure your answer as a readable text with key findings, not as JSON or bullet points. "
        "If there are notable examples, mention 2-3 of them briefly. "
        "If there are relevant links, mention them. "
        "Be concise but informative. Do not repeat the question back."
    )
    payload = {
        "question": query,
        "query_type": query_type,
        "data_used": data_used,
        "comparison": comparison,
        "evaluation_metric": metric,
        "limitations_and_confidence": limitations,
        "examples": examples[:TOP_EXAMPLES_K],
        "links": list(links[:TOP_EXAMPLES_K]),
    }
    user_prompt = (
        f"Answer the following procurement question based on the data below.\n\n"
        f"Question: {query}\n\n"
        f"Data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _llm_timeout(config: Mapping[str, object]) -> int:
    llm_cfg = _as_mapping(config.get("llm"))
    timeout = _to_int(llm_cfg.get("timeout") if llm_cfg is not None else 30, default=30)
    return timeout if timeout > 0 else 30


def _llm_model(config: Mapping[str, object]) -> str | None:
    llm_cfg = _as_mapping(config.get("llm"))
    if llm_cfg is None:
        return None
    model = _to_text(llm_cfg.get("model"))
    return model or None


def _rewrite_sections_with_llm(
    query: str,
    language: Language,
    query_type: SupportedQueryType,
    data_used: DataUsedSection,
    comparison: ComparisonSection,
    metric: EvaluationMetricSection,
    limitations: LimitationsConfidenceSection,
    examples: Sequence[ExampleItem],
    config: Mapping[str, object],
) -> _LLMSectionsModel | None:
    try:
        timeout = _llm_timeout(config)
        model = _llm_model(config)
        client = LLMClient(timeout=timeout, model=model)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("llm_response_rewrite_unavailable error=%s", type(error).__name__)
        return None

    try:
        response = client.chat(
            messages=_build_llm_messages(
                query,
                language,
                query_type,
                data_used,
                comparison,
                metric,
                limitations,
                examples,
            ),
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        parsed = cast(object, json.loads(response))
        if not isinstance(parsed, Mapping):
            return None
        parsed_map = cast(Mapping[str, object], parsed)
        return _LLMSectionsModel.model_validate(dict(parsed_map))
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        LOGGER.warning("llm_response_rewrite_invalid error=%s", type(error).__name__)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("llm_response_rewrite_failed error=%s", type(error).__name__)
    return None


def _generate_natural_language_answer(
    query: str,
    language: Language,
    query_type: SupportedQueryType,
    data_used: DataUsedSection,
    comparison: ComparisonSection,
    metric: EvaluationMetricSection,
    limitations: LimitationsConfidenceSection,
    examples: Sequence[ExampleItem],
    links: Sequence[str],
    config: Mapping[str, object],
    *,
    fallback_summary: str,
) -> str:
    try:
        timeout = _llm_timeout(config)
        model = _llm_model(config)
        client = LLMClient(timeout=timeout, model=model)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("llm_answer_generation_unavailable error=%s", type(error).__name__)
        return fallback_summary

    try:
        response = client.chat(
            messages=_build_answer_llm_messages(
                query,
                language,
                query_type,
                data_used,
                comparison,
                metric,
                limitations,
                examples,
                links,
            ),
            temperature=0.3,
            max_tokens=1500,
        )
        answer = response.strip()
        if answer:
            return answer
        return fallback_summary
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("llm_answer_generation_failed error=%s", type(error).__name__)
        return fallback_summary


def _generate_comparison_answer(
    query: str,
    language: Language,
    per_period: list[dict[str, object]],
    _chunks: Sequence[RetrievedChunk],
    config: Mapping[str, object],
) -> str:
    period_summaries: list[str] = []
    for entry in per_period:
        line = (
            f"{entry['period']}: {entry['contracts']} contracts, "
            + f"weighted avg {entry.get('weighted_average', 'N/A')}, "
            + f"median {entry.get('median', 'N/A')}, "
            + f"{entry.get('outliers_count', 0)} outlier(s)"
        )
        period_summaries.append(line)
    fallback = "Comparison by period:\n" + "\n".join(period_summaries)

    try:
        timeout = _llm_timeout(config)
        model = _llm_model(config)
        client = LLMClient(timeout=timeout, model=model)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning(
            "llm_comparison_answer_unavailable error=%s",
            type(error).__name__,
        )
        return fallback

    lang_name = _llm_language_name(language)

    period_block = ""
    for entry in per_period:
        outlier_lines = ""
        outliers = entry.get("outliers", [])
        if isinstance(outliers, list):
            for outlier in outliers:
                if isinstance(outlier, dict):
                    outlier_lines += (
                        f"    - procurement {outlier.get('procurement_id')}: "
                        f"value={outlier.get('value')}, z_score={outlier.get('z_score')}\n"
                    )
        period_block += (
            f"Period: {entry['period']}\n"
            f"  Contracts analysed: {entry['contracts']}\n"
            f"  Weighted average: {entry.get('weighted_average', 'N/A')}\n"
            f"  Median: {entry.get('median', 'N/A')}\n"
            f"  Outliers ({entry.get('outliers_count', 0)}):\n{outlier_lines}\n"
        )

    system_prompt = (
        "You are a procurement-analytics assistant. "
        "The user asked a COMPARISON question about public procurement data. "
        "You are given per-period statistics below. "
        "Write a clear, concise side-by-side comparison highlighting "
        "key differences and trends between the periods. "
        "Include numbers. Do NOT invent data — use only what is provided. "
        f"Answer in {lang_name}."
    )

    user_prompt = (
        f"User question: {query}\n\n"
        f"Per-period data:\n{period_block}\n"
        "Provide a concise comparison answer."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat(
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )
        answer = response.strip()
        if answer:
            return answer
        return fallback
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("llm_comparison_answer_failed error=%s", type(error).__name__)
        return fallback


def _compose_sections(
    *,
    query: str,
    query_type: SupportedQueryType,
    language: Language,
    entities: Entities,
    chunks: Sequence[RetrievedChunk],
    analytics_payload: Mapping[str, object],
    period_text: str,
    config: Mapping[str, object],
) -> tuple[ResponseSections, ExplainabilityMetadata, str]:
    contracts_analyzed = _to_int(analytics_payload.get("contracts_analyzed"), default=0)
    if contracts_analyzed == 0:
        contracts_analyzed = sum(1 for chunk in chunks if chunk["entity_type"] == "contract")

    if contracts_analyzed > 0:
        data_completeness = 0.8
    else:
        data_completeness = _calculate_data_completeness(chunks)

    confidence = _calculate_confidence(
        contracts_analyzed,
        entities,
        chunks=chunks,
        query_type=query_type,
        data_completeness=data_completeness,
    )
    methodology = "hybrid_retrieval + analytics_queries + deterministic_scoring"

    data_used: DataUsedSection = {
        "period": period_text,
        "filters": {
            "bins": entities.get("bins", []),
            "enstr_codes": entities.get("enstr_codes", []),
            "kato_codes": [],
        },
        "entities": {
            "contracts": contracts_analyzed,
            "lots": sum(1 for chunk in chunks if chunk["entity_type"] == "lot"),
            "plans": sum(1 for chunk in chunks if chunk["entity_type"] == "plan"),
        },
        "sample_size": contracts_analyzed,
    }
    comparison = _build_comparison_section(chunks, analytics_payload, query_type=query_type)
    metric = _metric_for_query_type(query_type)

    yoy_rows = analytics_payload.get("year_over_year_trends") if analytics_payload else None
    if isinstance(yoy_rows, list) and metric.get("metric_name") == "Year-over-year change":
        for row in reversed(yoy_rows):
            if isinstance(row, (list, tuple)) and len(row) >= 4 and row[3] is not None:
                metric["value"] = float(row[3])
                break

    temporal_coverage = "full period" if len(entities.get("dates", [])) <= 1 else "partial period"
    limitations_list: list[str] = []
    if contracts_analyzed < 5:
        limitations_list.append("Small sample size (n < 5) lowers statistical confidence.")
    if contracts_analyzed < 20:
        limitations_list.append(
            f"Analysis based on {contracts_analyzed} contracts - larger sample would improve confidence."
        )
    if data_completeness < 0.8:
        limitations_list.append("Some records have missing key fields (e.g., sign_date/amount).")
    if not limitations_list:
        limitations_list.append(
            "No critical data quality constraints detected in retrieved sample."
        )

    limitations: LimitationsConfidenceSection = {
        "limitations": limitations_list,
        "data_completeness": round(data_completeness, 3),
        "temporal_coverage": temporal_coverage,
        "confidence": confidence,
    }
    examples = _build_examples(chunks)
    links = _generate_links(chunks, config)

    llm_rewrite = _rewrite_sections_with_llm(
        query,
        language,
        query_type,
        data_used,
        comparison,
        metric,
        limitations,
        examples,
        config,
    )

    summary = _default_summary(query_type, contracts_analyzed)
    comparison_text = comparison["methodology"]
    metric_text = metric["formula"]
    limitations_text = "; ".join(limitations["limitations"])
    if llm_rewrite is not None:
        summary = llm_rewrite.summary.strip() or summary
        comparison_text = llm_rewrite.comparison.strip() or comparison_text
        metric_text = llm_rewrite.evaluation_metric.strip() or metric_text
        limitations_text = llm_rewrite.limitations_and_confidence.strip() or limitations_text

    sections: ResponseSections = {
        "summary": summary,
        "data_used": data_used,
        "comparison": {
            **comparison,
            "methodology": comparison_text,
        },
        "evaluation_metric": {
            **metric,
            "formula": metric_text,
        },
        "limitations_and_confidence": {
            **limitations,
            "limitations": [limitations_text],
        },
        "examples": examples,
        "links": links,
    }

    metadata: ExplainabilityMetadata = {
        "sample_size": contracts_analyzed,
        "methodology": methodology,
        "confidence": confidence,
        "comparison_methodology": "weighted averages + medians + outlier z-score (>2σ)",
        "query_type": query_type,
        "language": language,
    }

    answer = _generate_natural_language_answer(
        query,
        language,
        query_type,
        data_used,
        comparison,
        metric,
        limitations,
        examples,
        links,
        config,
        fallback_summary=summary,
    )

    return sections, metadata, answer


def _run_pipeline_for_query_type(
    query: str,
    classification: ClassificationResult,
    *,
    config: Mapping[str, object],
) -> GeneratedResponse:
    query_type = classification["query_type"]
    language = classification["language"]
    entities = classification["entities"]

    start, end, period_text = _resolve_period(entities, config)
    retrieval_filters = _build_retrieval_filters(entities, start, end)
    chunks = _retrieve_context(query, retrieval_filters, query_type=query_type)

    bins = [_to_int(item, default=0) for item in entities.get("bins", [])]
    bins = [value for value in bins if value > 0]
    analytics_payload = _safe_analytics(bins, entities.get("enstr_codes", []), start, end)

    sections, metadata, answer = _compose_sections(
        query=query,
        query_type=query_type,
        language=language,
        entities=entities,
        chunks=chunks,
        analytics_payload=analytics_payload,
        period_text=period_text,
        config=config,
    )

    return {
        "query": query,
        "query_type": query_type,
        "language": language,
        "answer": answer,
        "sections": sections,
        "metadata": metadata,
    }


def _handle_search(
    query: str,
    classification: ClassificationResult,
    config: Mapping[str, object],
) -> GeneratedResponse:
    return _run_pipeline_for_query_type(query, classification, config=config)


def _handle_comparison(
    query: str,
    classification: ClassificationResult,
    config: Mapping[str, object],
) -> GeneratedResponse:
    entities = classification["entities"]
    periods = _resolve_individual_periods(entities, config)

    if len(periods) < 2:
        return _run_pipeline_for_query_type(query, classification, config=config)

    query_type: SupportedQueryType = "COMPARISON"
    language = classification["language"]

    bins = [_to_int(item, default=0) for item in entities.get("bins", [])]
    bins = [value for value in bins if value > 0]

    all_chunks: list[RetrievedChunk] = []
    per_period: list[dict[str, object]] = []

    for start, end, label in periods:
        filters = _build_retrieval_filters(entities, start, end)
        chunks = _retrieve_context(query, filters, query_type=query_type)
        analytics = _safe_analytics(bins, entities.get("enstr_codes", []), start, end)
        contracts_count = _to_int(analytics.get("contracts_analyzed"), default=0)
        if contracts_count == 0:
            contracts_count = sum(1 for c in chunks if c["entity_type"] == "contract")
        comparison = _build_comparison_section(chunks, analytics, query_type=query_type)
        per_period.append(
            {
                "period": label,
                "contracts": contracts_count,
                "weighted_average": comparison["weighted_average"],
                "median": comparison["median"],
                "outliers_count": len(comparison["outliers"]),
                "outliers": comparison["outliers"],
            }
        )
        all_chunks.extend(chunks)

    combined_start = periods[0][0]
    combined_end = periods[-1][1]
    combined_period = f"{combined_start.date().isoformat()} - {combined_end.date().isoformat()}"
    combined_analytics = _safe_analytics(
        bins, entities.get("enstr_codes", []), combined_start, combined_end
    )

    sections, metadata, _ = _compose_sections(
        query=query,
        query_type=query_type,
        language=language,
        entities=entities,
        chunks=all_chunks,
        analytics_payload=combined_analytics,
        period_text=combined_period,
        config=config,
    )

    answer = _generate_comparison_answer(
        query,
        language,
        per_period,
        all_chunks,
        config,
    )

    return {
        "query": query,
        "query_type": query_type,
        "language": language,
        "answer": answer,
        "sections": sections,
        "metadata": metadata,
    }


def _handle_analytics(
    query: str,
    classification: ClassificationResult,
    config: Mapping[str, object],
) -> GeneratedResponse:
    return _run_pipeline_for_query_type(query, classification, config=config)


def _handle_anomaly_detection(
    query: str,
    classification: ClassificationResult,
    config: Mapping[str, object],
) -> GeneratedResponse:
    return _run_pipeline_for_query_type(query, classification, config=config)


def _handle_fairness(
    query: str,
    classification: ClassificationResult,
    config: Mapping[str, object],
) -> GeneratedResponse:
    return _run_pipeline_for_query_type(query, classification, config=config)


def _handle_fallback(
    query: str,
    classification: ClassificationResult,
    config: Mapping[str, object],
) -> GeneratedResponse:
    return _run_pipeline_for_query_type(query, classification, config=config)


def generate_response(query: str, config_path: Path = DEFAULT_CONFIG_PATH) -> GeneratedResponse:
    config = _load_config(config_path)
    classification = classify(query, config_path=config_path)
    query_type = classification["query_type"]

    LOGGER.info(
        "response_generation_started query_length=%d query_type=%s language=%s",
        len(query),
        query_type,
        classification["language"],
    )

    if query_type == "SEARCH":
        return _handle_search(query, classification, config)
    if query_type == "COMPARISON":
        return _handle_comparison(query, classification, config)
    if query_type == "ANALYTICS":
        return _handle_analytics(query, classification, config)
    if query_type == "ANOMALY_DETECTION":
        return _handle_anomaly_detection(query, classification, config)
    if query_type == "FAIRNESS":
        return _handle_fairness(query, classification, config)
    return _handle_fallback(query, classification, config)


def generate(query: str, config_path: Path = DEFAULT_CONFIG_PATH) -> GeneratedResponse:
    return generate_response(query, config_path=config_path)


def _format_markdown(response: GeneratedResponse) -> str:
    sections = response["sections"]
    data_used = sections["data_used"]
    comparison = sections["comparison"]
    metric = sections["evaluation_metric"]
    limits = sections["limitations_and_confidence"]

    lines: list[str] = []
    lines.append("1. Summary")
    lines.append(sections["summary"])
    lines.append("")
    lines.append("2. Data used")
    lines.append(f"- Period: {data_used['period']}")
    lines.append(f"- Filters: {json.dumps(data_used['filters'], ensure_ascii=False)}")
    lines.append(f"- Entities: {json.dumps(data_used['entities'], ensure_ascii=False)}")
    lines.append(f"- Sample size: N={data_used['sample_size']}")
    lines.append("")
    lines.append("3. Comparison")
    lines.append(
        f"- Weighted average: {comparison['weighted_average']} | Median: {comparison['median']}"
    )
    lines.append(f"- Outliers: {json.dumps(comparison['outliers'], ensure_ascii=False)}")
    lines.append(f"- Baseline: {comparison['reference_baseline']}")
    lines.append(f"- Methodology: {comparison['methodology']}")
    lines.append("")
    lines.append("4. Evaluation Metric")
    lines.append(f"- Metric: {metric['metric_name']}")
    lines.append(f"- Formula: {metric['formula']}")
    lines.append(f"- Threshold: {metric['threshold']}")
    lines.append(f"- Value: {metric['value']}")
    lines.append("")
    lines.append("5. Limitations and Confidence")
    lines.append(f"- Limitations: {json.dumps(limits['limitations'], ensure_ascii=False)}")
    lines.append(f"- Data completeness: {limits['data_completeness']}")
    lines.append(f"- Temporal coverage: {limits['temporal_coverage']}")
    lines.append(f"- Confidence: {limits['confidence']}")
    lines.append("")
    lines.append("6. Examples (top-k)")
    for example in sections["examples"]:
        lines.append(f"- {json.dumps(example, ensure_ascii=False)}")
    lines.append("")
    lines.append("7. Links")
    for link in sections["links"]:
        lines.append(f"- {link}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Procurement response generator")
    _ = parser.add_argument("--query", type=str, required=True, help="Query text")
    _ = parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    _ = parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format",
    )
    _ = parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        stdout_buffer = cast(_StdoutBuffer | None, getattr(sys.stdout, "buffer", None))
        if stdout_buffer is not None:
            _ = stdout_buffer.write(text.encode("utf-8", errors="replace"))
            _ = stdout_buffer.write(b"\n")
            stdout_buffer.flush()
            return
        print(text.encode("ascii", errors="replace").decode("ascii"))


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args_map = cast(dict[str, object], vars(args))

    logging.basicConfig(
        level=logging.DEBUG if bool(args_map.get("verbose")) else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    query_text = _to_text(args_map.get("query"))
    config_path = cast(Path, args_map.get("config", DEFAULT_CONFIG_PATH))
    output_format = _to_text(args_map.get("format"), default="json")

    try:
        response = generate_response(query_text, config_path=config_path)
        if output_format == "markdown":
            _safe_print(_format_markdown(response))
        else:
            _safe_print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001
        LOGGER.error("response_generation_failed error=%s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
