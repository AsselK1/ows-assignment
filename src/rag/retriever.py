from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_MAX_TOKENS = 4000
DEFAULT_RERANK_TOP_K = 20
DEFAULT_BM25_CANDIDATES = 200
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ClickHouseClient(Protocol):
    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]: ...

    def disconnect(self) -> None: ...


class _VectorSearchResult(Protocol):
    procurement_id: int
    entity_type: str
    content: str
    score: float
    metadata: dict[str, object]


class _VectorStore(Protocol):
    def search(
        self,
        query: str,
        filters: Mapping[str, object] | None = None,
        limit: int = 10,
    ) -> list[_VectorSearchResult]: ...


class _CrossEncoder(Protocol):
    def predict(
        self,
        sentences: Sequence[Sequence[str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class _KeywordCandidate:
    procurement_id: int
    entity_type: str
    content: str
    customer_bin: int
    enstr_code: str
    sign_date: str
    contract_sum: float
    supplier_bin: int
    metadata: dict[str, object]
    bm25_score: float


@dataclass
class _HybridCandidate:
    procurement_id: int
    entity_type: str
    content: str
    metadata: dict[str, object]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    score: float = 0.0


def _module_attr(module: ModuleType, name: str) -> object:
    return cast(object, getattr(module, name))


def _build_instance(factory: object, *args: object, **kwargs: object) -> object:
    return cast(Callable[..., object], factory)(*args, **kwargs)


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
    try:
        return int(text)
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


def _to_text(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _to_iso_datetime(value: object) -> str:
    if isinstance(value, datetime):
        normalized = value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value
        return normalized.isoformat(timespec="seconds")

    text = _to_text(value)
    if not text:
        return ""
    normalized_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized_text)
        normalized = parsed.astimezone(UTC).replace(tzinfo=None) if parsed.tzinfo else parsed
        return normalized.isoformat(timespec="seconds")
    except ValueError:
        return text


def _tokenize(text: str) -> list[str]:
    raw_tokens = cast(list[str], re.findall(r"[\w-]+", text, flags=re.UNICODE))
    tokens = [item.lower() for item in raw_tokens]
    return [token for token in tokens if len(token) > 1]


def _normalize_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if math.isclose(max_value, min_value):
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def _truncate_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]).strip()


def _extract_customer_bins(filters: Mapping[str, object] | None) -> tuple[int, ...]:
    if not filters:
        return ()
    customer_bin = filters.get("customer_bin")
    if customer_bin is None:
        return ()
    if isinstance(customer_bin, Sequence) and not isinstance(customer_bin, (str, bytes, bytearray)):
        values = tuple(sorted({_to_int(item) for item in customer_bin if _to_int(item) > 0}))
        return values
    parsed = _to_int(customer_bin)
    return (parsed,) if parsed > 0 else ()


def _get_clickhouse_client() -> ClickHouseClient:
    import clickhouse_driver

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

    secure = port in (9440, 8443)
    return cast(
        ClickHouseClient,
        clickhouse_driver.Client(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            secure=secure,
            connect_timeout=15,
            send_receive_timeout=30,
        ),
    )


def _build_vector_store(qdrant_host: str, qdrant_port: int) -> _VectorStore:
    vector_store_module = importlib.import_module("src.rag.vector_store")
    vector_store_factory = _module_attr(vector_store_module, "VectorStore")
    return cast(
        _VectorStore,
        _build_instance(vector_store_factory, qdrant_host=qdrant_host, qdrant_port=qdrant_port),
    )


def _query_keyword_candidates(
    clickhouse_client: ClickHouseClient,
    query: str,
    filters: Mapping[str, object] | None,
    limit: int,
) -> list[_KeywordCandidate]:
    terms = _tokenize(query)
    if not terms:
        return []

    has_date_filter = bool(
        _to_iso_datetime(filters.get("date_from") if filters else None)
        or _to_iso_datetime(filters.get("date_to") if filters else None)
    )
    has_bin_filter = bool(_extract_customer_bins(filters))
    # When structural filters (date/bin) narrow the result set, skip the text-match
    # requirement — the content strings are English-only and won't match Russian queries.
    if has_date_filter or has_bin_filter:
        where_clauses: list[str] = []
    else:
        where_clauses = [
            "(multiSearchAnyCaseInsensitiveUTF8(content, %(terms)s) = 1 OR positionCaseInsensitiveUTF8(content, %(query_text)s) > 0)"
        ]
    params: dict[str, object] = {
        "terms": terms,
        "query_text": query,
        "limit": max(1, limit),
    }

    customer_bins = _extract_customer_bins(filters)
    if customer_bins:
        where_clauses.append("customer_bin IN %(customer_bins)s")
        params["customer_bins"] = customer_bins

    enstr_code = _to_text(filters.get("enstr_code") if filters else None)
    if enstr_code:
        where_clauses.append("positionCaseInsensitiveUTF8(enstr_code, %(enstr_code)s) > 0")
        params["enstr_code"] = enstr_code

    date_from = _to_iso_datetime(filters.get("date_from") if filters else None)
    if date_from:
        where_clauses.append("sign_date >= toDateTime(%(date_from)s)")
        params["date_from"] = date_from

    date_to = _to_iso_datetime(filters.get("date_to") if filters else None)
    if date_to:
        where_clauses.append("sign_date <= toDateTime(%(date_to)s)")
        params["date_to"] = date_to

    sql = f"""
    SELECT
        procurement_id,
        entity_type,
        content,
        customer_bin,
        enstr_code,
        sign_date,
        sum_value,
        sup_bin
    FROM (
        SELECT
            id AS procurement_id,
            'contract' AS entity_type,
            concat(
                'Contract ',
                argMax(contract_number, updated_at),
                ' ENSTR ',
                argMax(enstr_code_lvl4, updated_at),
                ' KATO ',
                argMax(kato_code, updated_at),
                ' amount ',
                toString(argMax(contract_sum, updated_at))
            ) AS content,
            toInt64(argMax(customer_bin, updated_at)) AS customer_bin,
            toString(argMax(enstr_code_lvl4, updated_at)) AS enstr_code,
            toDateTime(argMax(sign_date, updated_at)) AS sign_date,
            toFloat64(argMax(contract_sum, updated_at)) AS sum_value,
            toInt64(argMax(supplier_bin, updated_at)) AS sup_bin
        FROM contracts
        GROUP BY id

        UNION ALL

        SELECT
            id AS procurement_id,
            'lot' AS entity_type,
            concat(
                'Lot ',
                argMax(lot_number, updated_at),
                ' ENSTR ',
                argMax(enstr_code_lvl4, updated_at),
                ' KATO ',
                argMax(kato_code, updated_at),
                ' amount ',
                toString(argMax(amount, updated_at))
            ) AS content,
            toInt64(argMax(customer_bin, updated_at)) AS customer_bin,
            toString(argMax(enstr_code_lvl4, updated_at)) AS enstr_code,
            toDateTime(argMax(created_at, updated_at)) AS sign_date,
            toFloat64(argMax(amount, updated_at)) AS sum_value,
            toInt64(0) AS sup_bin
        FROM lots
        GROUP BY id

        UNION ALL

        SELECT
            id AS procurement_id,
            'announcement' AS entity_type,
            concat(
                'Announcement ',
                argMax(number_anno, updated_at),
                ' total_sum ',
                toString(argMax(total_sum, updated_at)),
                ' lots ',
                toString(argMax(count_lots, updated_at))
            ) AS content,
            toInt64(argMax(customer_bin, updated_at)) AS customer_bin,
            '' AS enstr_code,
            toDateTime(argMax(publish_date, updated_at)) AS sign_date,
            toFloat64(argMax(total_sum, updated_at)) AS sum_value,
            toInt64(0) AS sup_bin
        FROM announcements
        GROUP BY id

        UNION ALL

        SELECT
            id AS procurement_id,
            'plan' AS entity_type,
            concat(
                'Plan ',
                argMax(plan_number, updated_at),
                ' ENSTR ',
                argMax(enstr_code_lvl4, updated_at),
                ' KATO ',
                argMax(kato_code, updated_at),
                ' amount ',
                toString(argMax(planned_amount, updated_at))
            ) AS content,
            toInt64(argMax(customer_bin, updated_at)) AS customer_bin,
            toString(argMax(enstr_code_lvl4, updated_at)) AS enstr_code,
            toDateTime(argMax(publish_date, updated_at)) AS sign_date,
            toFloat64(argMax(planned_amount, updated_at)) AS sum_value,
            toInt64(0) AS sup_bin
        FROM plans
        GROUP BY id
    ) AS docs
    WHERE {" AND ".join(where_clauses) if where_clauses else "1 = 1"}
    ORDER BY sign_date DESC
    LIMIT %(limit)s
    """

    rows = clickhouse_client.execute(sql, params)
    if not rows:
        return []

    doc_tokens = [_tokenize(_to_text(row[2])) for row in rows]
    doc_lengths = [max(1, len(tokens)) for tokens in doc_tokens]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    document_frequency: dict[str, int] = {
        term: sum(1 for tokens in doc_tokens if term in set(tokens)) for term in terms
    }

    k1 = 1.5
    b = 0.75
    doc_boost = {"contract": 1.0, "lot": 0.95, "announcement": 0.9, "plan": 0.85}
    total_docs = len(rows)

    candidates: list[_KeywordCandidate] = []
    for row, tokens, doc_len in zip(rows, doc_tokens, doc_lengths, strict=False):
        procurement_id = _to_int(row[0])
        entity_type = _to_text(row[1])
        content = _to_text(row[2])
        customer_bin = _to_int(row[3])
        enstr = _to_text(row[4])
        sign_date = _to_iso_datetime(row[5])
        contract_sum = _to_float(row[6], default=0.0)
        supplier_bin = _to_int(row[7])

        score = 0.0
        for term in terms:
            frequency = tokens.count(term)
            if frequency <= 0:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log1p((total_docs - df + 0.5) / (df + 0.5))
            norm = (frequency * (k1 + 1.0)) / (
                frequency + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_length)))
            )
            score += math.log1p(float(frequency)) * idf * norm
        score *= doc_boost.get(entity_type, 0.8)

        metadata: dict[str, object] = {
            "customer_bin": customer_bin,
            "enstr_code": enstr,
            "sign_date": sign_date,
            "contract_sum": contract_sum,
            "supplier_bin": supplier_bin,
            "chunk_index": 0,
            "source": "bm25",
        }
        candidates.append(
            _KeywordCandidate(
                procurement_id=procurement_id,
                entity_type=entity_type,
                content=content,
                customer_bin=customer_bin,
                enstr_code=enstr,
                sign_date=sign_date,
                contract_sum=contract_sum,
                supplier_bin=supplier_bin,
                metadata=metadata,
                bm25_score=score,
            )
        )

    return candidates


def _rerank_candidates(
    query: str,
    candidates: list[_HybridCandidate],
    rerank_top_k: int,
) -> dict[tuple[str, int, str], float]:
    if not candidates:
        return {}

    sentence_transformers_module = importlib.import_module("sentence_transformers")
    cross_encoder_factory = _module_attr(sentence_transformers_module, "CrossEncoder")
    cross_encoder = cast(
        _CrossEncoder,
        _build_instance(cross_encoder_factory, DEFAULT_CROSS_ENCODER_MODEL),
    )

    top_candidates = sorted(candidates, key=lambda item: item.fused_score, reverse=True)[
        :rerank_top_k
    ]
    pairs = [[query, candidate.content] for candidate in top_candidates]
    predicted = cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
    return {
        (candidate.entity_type, candidate.procurement_id, candidate.content): _to_float(score)
        for candidate, score in zip(top_candidates, predicted, strict=False)
    }


def _fuse_candidates(
    query: str,
    vector_results: Sequence[_VectorSearchResult],
    keyword_results: Sequence[_KeywordCandidate],
    *,
    vector_weight: float,
    rerank_top_k: int,
) -> list[_HybridCandidate]:
    by_key: dict[tuple[str, int, str], _HybridCandidate] = {}

    for vector_item in vector_results:
        key = (vector_item.entity_type, vector_item.procurement_id, vector_item.content)
        candidate = by_key.get(key)
        if candidate is None:
            by_key[key] = _HybridCandidate(
                procurement_id=vector_item.procurement_id,
                entity_type=vector_item.entity_type,
                content=vector_item.content,
                metadata=dict(vector_item.metadata),
                vector_score=float(vector_item.score),
            )
        else:
            candidate.vector_score = max(candidate.vector_score, float(vector_item.score))

    for keyword_item in keyword_results:
        key = (keyword_item.entity_type, keyword_item.procurement_id, keyword_item.content)
        candidate = by_key.get(key)
        if candidate is None:
            by_key[key] = _HybridCandidate(
                procurement_id=keyword_item.procurement_id,
                entity_type=keyword_item.entity_type,
                content=keyword_item.content,
                metadata=dict(keyword_item.metadata),
                bm25_score=keyword_item.bm25_score,
            )
        else:
            candidate.bm25_score = max(candidate.bm25_score, keyword_item.bm25_score)
            merged_metadata = dict(candidate.metadata)
            merged_metadata.update(keyword_item.metadata)
            candidate.metadata = merged_metadata

    candidates = list(by_key.values())
    vector_norm = _normalize_scores([candidate.vector_score for candidate in candidates])
    bm25_norm = _normalize_scores([candidate.bm25_score for candidate in candidates])

    for candidate, vector_score_norm, bm25_score_norm in zip(
        candidates, vector_norm, bm25_norm, strict=False
    ):
        candidate.fused_score = (
            max(0.0, min(1.0, vector_weight)) * vector_score_norm
            + (1.0 - max(0.0, min(1.0, vector_weight))) * bm25_score_norm
        )
        candidate.score = candidate.fused_score

    rerank_raw: dict[tuple[str, int, str], float] = {}
    if candidates:
        try:
            rerank_raw = _rerank_candidates(query, candidates, rerank_top_k=rerank_top_k)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("cross_encoder_rerank_failed error=%s", error)

    if rerank_raw:
        rerank_keys = list(rerank_raw.keys())
        rerank_values = _normalize_scores(list(rerank_raw.values()))
        rerank_normalized = {
            key: score for key, score in zip(rerank_keys, rerank_values, strict=False)
        }
        for candidate in candidates:
            key = (candidate.entity_type, candidate.procurement_id, candidate.content)
            candidate.rerank_score = rerank_normalized.get(key, 0.0)
            candidate.score = (0.6 * candidate.rerank_score) + (0.4 * candidate.fused_score)

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _enforce_context_window(
    candidates: Sequence[_HybridCandidate],
    *,
    limit: int,
    max_tokens: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    used_tokens = 0

    for candidate in candidates:
        if len(selected) >= limit:
            break
        truncated_content = _truncate_tokens(candidate.content, max_tokens=350)
        chunk_tokens = len(truncated_content.split())
        if chunk_tokens <= 0:
            continue
        if used_tokens + chunk_tokens > max_tokens:
            continue

        metadata = dict(candidate.metadata)
        _ = metadata.setdefault("entity_type", candidate.entity_type)
        _ = metadata.setdefault("procurement_id", candidate.procurement_id)
        _ = metadata.setdefault("chunk_index", _to_int(metadata.get("chunk_index"), default=0))
        metadata["vector_score"] = candidate.vector_score
        metadata["bm25_score"] = candidate.bm25_score
        metadata["fused_score"] = candidate.fused_score
        metadata["rerank_score"] = candidate.rerank_score
        metadata["token_count"] = chunk_tokens

        selected.append(
            {
                "content": truncated_content,
                "entity_type": candidate.entity_type,
                "procurement_id": candidate.procurement_id,
                "score": candidate.score,
                "metadata": metadata,
            }
        )
        used_tokens += chunk_tokens

    return selected


class HybridRetriever:
    _vector_store: _VectorStore
    _clickhouse: ClickHouseClient | None
    vector_weight: float
    bm25_candidates: int
    rerank_top_k: int

    def __init__(
        self,
        *,
        qdrant_host: str = DEFAULT_QDRANT_HOST,
        qdrant_port: int = DEFAULT_QDRANT_PORT,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        bm25_candidates: int = DEFAULT_BM25_CANDIDATES,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> None:
        self._vector_store = _build_vector_store(qdrant_host=qdrant_host, qdrant_port=qdrant_port)
        self._clickhouse = None
        try:
            self._clickhouse = _get_clickhouse_client()
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("clickhouse_unavailable_for_bm25 error=%s", error)
        self.vector_weight = vector_weight
        self.bm25_candidates = bm25_candidates
        self.rerank_top_k = rerank_top_k

    def close(self) -> None:
        if self._clickhouse is not None:
            self._clickhouse.disconnect()

    def retrieve(
        self,
        query: str,
        filters: Mapping[str, object] | None = None,
        limit: int = 10,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> list[dict[str, object]]:
        if not query.strip():
            return []

        vector_results: list[_VectorSearchResult] = []
        try:
            vector_results = self._vector_store.search(
                query, filters=filters, limit=max(limit * 2, 20)
            )
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("vector_search_failed error=%s", error)

        keyword_results: list[_KeywordCandidate] = []
        if self._clickhouse is not None:
            try:
                keyword_results = _query_keyword_candidates(
                    self._clickhouse,
                    query,
                    filters,
                    limit=max(self.bm25_candidates, limit * 5),
                )
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("bm25_keyword_search_failed error=%s", error)

        fused = _fuse_candidates(
            query,
            vector_results,
            keyword_results,
            vector_weight=self.vector_weight,
            rerank_top_k=self.rerank_top_k,
        )
        return _enforce_context_window(fused, limit=limit, max_tokens=max_tokens)


def retrieve(
    query: str,
    filters: dict[str, object] | None = None,
    limit: int = 10,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    qdrant_host: str = DEFAULT_QDRANT_HOST,
    qdrant_port: int = DEFAULT_QDRANT_PORT,
) -> list[dict[str, object]]:
    retriever = HybridRetriever(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        vector_weight=vector_weight,
    )
    try:
        return retriever.retrieve(query, filters=filters, limit=limit, max_tokens=max_tokens)
    finally:
        retriever.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retriever for procurement RAG")
    _query_action = parser.add_argument("--query", type=str, default="", help="Query text")
    _filters_action = parser.add_argument(
        "--filters",
        type=str,
        default="",
        help=(
            "JSON filters (customer_bin, enstr_code, date_from, date_to), "
            'for example: {"customer_bin": [123], "date_from": "2025-01-01"}'
        ),
    )
    _limit_action = parser.add_argument("--limit", type=int, default=10, help="Result chunk limit")
    _max_tokens_action = parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum context tokens to return",
    )
    _vector_weight_action = parser.add_argument(
        "--vector-weight",
        type=float,
        default=DEFAULT_VECTOR_WEIGHT,
        help="Weight for vector score during fusion (0-1)",
    )
    _config_action = parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    _host_action = parser.add_argument(
        "--qdrant-host",
        type=str,
        default=DEFAULT_QDRANT_HOST,
        help="Qdrant host (default: localhost)",
    )
    _port_action = parser.add_argument(
        "--qdrant-port",
        type=int,
        default=DEFAULT_QDRANT_PORT,
        help="Qdrant port (default: 6333)",
    )
    _verbose_action = parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    _ = (
        _query_action,
        _filters_action,
        _limit_action,
        _max_tokens_action,
        _vector_weight_action,
        _config_action,
        _host_action,
        _port_action,
        _verbose_action,
    )
    args = parser.parse_args()
    args_map = cast(dict[str, object], vars(args))

    logging.basicConfig(
        level=logging.DEBUG if bool(args_map.get("verbose")) else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    qdrant_host = _to_text(args_map.get("qdrant_host"), default=DEFAULT_QDRANT_HOST)
    qdrant_port = _to_int(args_map.get("qdrant_port"), default=DEFAULT_QDRANT_PORT)

    config_path = cast(Path, args_map.get("config", DEFAULT_CONFIG_PATH))
    try:
        config = _load_config(config_path)
        qdrant_from_config = _as_mapping(config.get("qdrant"))
        if qdrant_from_config is not None:
            qdrant_host = _to_text(qdrant_from_config.get("host"), default=qdrant_host)
            qdrant_port = _to_int(qdrant_from_config.get("port"), default=qdrant_port)
    except Exception as error:  # noqa: BLE001
        LOGGER.debug("config_read_failed path=%s error=%s", config_path, error)

    filters: dict[str, object] | None = None
    filters_raw = _to_text(args_map.get("filters"))
    if filters_raw:
        parsed_obj = cast(object, json.loads(filters_raw))
        if isinstance(parsed_obj, dict):
            filters = cast(dict[str, object], parsed_obj)

    query = _to_text(args_map.get("query"))
    if not query:
        parser.print_help()
        return

    output = retrieve(
        query=query,
        filters=filters,
        limit=max(1, _to_int(args_map.get("limit"), default=10)),
        max_tokens=max(1, _to_int(args_map.get("max_tokens"), default=DEFAULT_MAX_TOKENS)),
        vector_weight=max(0.0, min(1.0, _to_float(args_map.get("vector_weight"), default=0.7))),
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )
    LOGGER.info("retrieval_completed results=%d", len(output))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
