from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
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

DEFAULT_COLLECTION = "procurement_documents"
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384


class ClickHouseClient(Protocol):
    def query(self, query: str, params: object | None = None) -> object: ...

    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]: ...

    def close(self) -> None: ...

    def disconnect(self) -> None: ...


class _Encoder(Protocol):
    def encode(
        self,
        text: str,
        *,
        convert_to_tensor: bool,
        show_progress_bar: bool,
    ) -> Sequence[float]: ...


class _CollectionDescription(Protocol):
    name: str


class _CollectionsResponse(Protocol):
    collections: Sequence[_CollectionDescription]


class _SearchHit(Protocol):
    payload: object
    score: float


class _QdrantClient(Protocol):
    def get_collections(self) -> _CollectionsResponse: ...

    def create_collection(self, *, collection_name: str, vectors_config: object) -> object: ...

    def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: object,
    ) -> object: ...

    def upsert(self, *, collection_name: str, points: Sequence[object]) -> object: ...

    def query_points(
        self,
        *,
        collection_name: str,
        query: Sequence[float],
        query_filter: object | None,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> object: ...


@dataclass(frozen=True)
class SearchResult:
    procurement_id: int
    entity_type: str
    content: str
    score: float
    metadata: dict[str, object]


def _module_attr(module: ModuleType, name: str) -> object:
    return cast(object, getattr(module, name))


def _build_instance(factory: object, *args: object, **kwargs: object) -> object:
    return cast(Callable[..., object], factory)(*args, **kwargs)


class _ClickHouseAdapter:
    def __init__(self, client):
        self._client = client

    def query(self, query: str, params: object | None = None) -> object:
        if params:
            query = query % params
        return self._client.query(query)

    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]:
        if params:
            query = query % params
        result = self._client.query(query)
        return result.result_rows if hasattr(result, "result_rows") else []

    def close(self) -> None:
        self._client.close()

    def disconnect(self) -> None:
        self._client.close()

    def disconnect(self) -> None:
        self._client.close()


def _get_clickhouse_client() -> ClickHouseClient:
    import clickhouse_connect

    host = os.getenv("CLICKHOUSE_HOST")
    if not host:
        raise ValueError("CLICKHOUSE_HOST is required")

    port_raw = os.getenv("CLICKHOUSE_PORT", "9000")
    try:
        port = int(port_raw)
    except ValueError as error:
        raise ValueError(f"CLICKHOUSE_PORT must be an integer, got: {port_raw}") from error

    database = os.getenv("CLICKHOUSE_DB")
    if not database:
        raise ValueError("CLICKHOUSE_DB is required")

    user = os.getenv("CLICKHOUSE_USER", "default")
    password = os.getenv("CLICKHOUSE_PASSWORD", "")

    secure = port in (8443, 9440)
    http_port = 8443 if port == 9440 else port
    client = clickhouse_connect.get_client(
        host=host,
        port=http_port,
        database=database,
        username=user,
        password=password,
        secure=secure,
        connect_timeout=15,
        send_receive_timeout=30,
    )
    return cast(ClickHouseClient, _ClickHouseAdapter(client))


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

    raw = _to_text(value)
    if not raw:
        return ""

    normalized_text = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized_text)
        normalized = parsed.astimezone(UTC).replace(tzinfo=None) if parsed.tzinfo else parsed
        return normalized.isoformat(timespec="seconds")
    except ValueError:
        return raw


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    if max_chars <= 0:
        return [normalized]
    if overlap < 0:
        overlap = 0
    if overlap >= max_chars:
        overlap = max(0, max_chars // 4)

    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", normalized) if item.strip()]
    if not sentences:
        sentences = [normalized]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def split_long(segment: str) -> None:
        stride = max(1, max_chars - overlap)
        start = 0
        while start < len(segment):
            end = min(start + max_chars, len(segment))
            piece = segment[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(segment):
                break
            start += stride

    for sentence in sentences:
        if len(sentence) > max_chars:
            flush()
            split_long(sentence)
            continue

        if not current:
            current = sentence
            continue

        candidate = f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        previous = current
        flush()
        prefix = previous[-overlap:] if overlap > 0 else ""
        current = f"{prefix} {sentence}".strip()
        if len(current) > max_chars:
            split_long(current)
            current = ""

    flush()
    return chunks


class VectorStore:
    _models: ModuleType
    _client: _QdrantClient
    _encoder: _Encoder
    collection_name: str

    def __init__(
        self,
        qdrant_host: str = DEFAULT_QDRANT_HOST,
        qdrant_port: int = DEFAULT_QDRANT_PORT,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        qdrant_client_module = importlib.import_module("qdrant_client")
        qdrant_models_module = importlib.import_module("qdrant_client.http.models")
        sentence_transformers_module = importlib.import_module("sentence_transformers")

        qdrant_client_factory = _module_attr(qdrant_client_module, "QdrantClient")
        encoder_factory = _module_attr(sentence_transformers_module, "SentenceTransformer")

        self._models = qdrant_models_module

        env_host = os.getenv("QDRANT_HOST", qdrant_host)
        env_api_key = os.getenv("QDRANT_API_KEY")

        client_kwargs = {}
        if env_api_key:
            client_kwargs["url"] = env_host
            client_kwargs["api_key"] = env_api_key
            client_kwargs["timeout"] = 10
            LOGGER.info("Connecting to Qdrant Cloud at %s", env_host)
        elif env_host.startswith("http://") or env_host.startswith("https://"):
            client_kwargs["url"] = env_host
            client_kwargs["timeout"] = 10
            LOGGER.info("Connecting to Qdrant at %s", env_host)
        else:
            client_kwargs["host"] = env_host
            client_kwargs["port"] = qdrant_port
            client_kwargs["timeout"] = 10
            LOGGER.info("Connecting to Qdrant at %s:%d", env_host, qdrant_port)

        self._client = cast(
            _QdrantClient,
            _build_instance(qdrant_client_factory, **client_kwargs),
        )
        self._encoder = cast(
            _Encoder,
            _build_instance(encoder_factory, DEFAULT_EMBEDDING_MODEL),
        )
        self.collection_name = collection_name

    def ensure_collection(self) -> None:
        collections = self._client.get_collections().collections
        existing = {item.name for item in collections}
        if self.collection_name not in existing:
            vector_params_factory = _module_attr(self._models, "VectorParams")
            distance = _module_attr(self._models, "Distance")
            cosine = _module_attr(cast(ModuleType, distance), "COSINE")
            vector_params = _build_instance(
                vector_params_factory, size=VECTOR_SIZE, distance=cosine
            )
            _ = self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params,
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        payload_schema_type = cast(ModuleType, _module_attr(self._models, "PayloadSchemaType"))
        integer_type = _module_attr(payload_schema_type, "INTEGER")
        keyword_type = _module_attr(payload_schema_type, "KEYWORD")
        datetime_type = _module_attr(payload_schema_type, "DATETIME")

        index_specs = (
            ("customer_bin", integer_type),
            ("supplier_bin", integer_type),
            ("enstr_code", keyword_type),
            ("sign_date", datetime_type),
        )
        for field_name, field_schema in index_specs:
            _ = self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

    def chunk_text(self, text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
        return chunk_text(text, max_chars=max_chars, overlap=overlap)

    def embed_text(self, text: str) -> list[float]:
        return [
            float(value)
            for value in self._encoder.encode(
                text,
                convert_to_tensor=False,
                show_progress_bar=False,
            )
        ]

    def _point_id(self, entity_type: str, procurement_id: int, chunk_index: int) -> int:
        payload = f"{entity_type}:{procurement_id}:{chunk_index}"
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _upsert_entity(
        self,
        *,
        entity_type: str,
        procurement_id: int,
        content: str,
        metadata: Mapping[str, object],
    ) -> int:
        chunks = self.chunk_text(content)
        if not chunks:
            return 0

        point_struct_factory = _module_attr(self._models, "PointStruct")
        points: list[object] = []
        total_chunks = len(chunks)

        customer_bin = _to_int(metadata.get("customer_bin"), default=0)
        supplier_bin_raw = metadata.get("supplier_bin")
        supplier_bin = (
            _to_int(supplier_bin_raw, default=0) if supplier_bin_raw is not None else None
        )

        for chunk_index, chunk in enumerate(chunks):
            point_id = self._point_id(entity_type, procurement_id, chunk_index)
            payload = {
                "procurement_id": procurement_id,
                "entity_type": entity_type,
                "content": chunk,
                "customer_bin": customer_bin,
                "supplier_bin": supplier_bin,
                "enstr_code": _to_text(metadata.get("enstr_code")),
                "kato_code": _to_text(metadata.get("kato_code")),
                "sign_date": _to_text(metadata.get("sign_date")),
                "contract_sum": _to_float(metadata.get("contract_sum"), default=0.0),
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "metadata": dict(metadata),
            }
            points.append(
                _build_instance(
                    point_struct_factory,
                    id=point_id,
                    vector=self.embed_text(chunk),
                    payload=payload,
                )
            )

        _ = self._client.upsert(collection_name=self.collection_name, points=points)
        return total_chunks

    def index_contract(self, contract_id: int, clickhouse_client: ClickHouseClient) -> int:
        rows = clickhouse_client.execute(
            """
            SELECT
                id,
                argMax(contract_number, updated_at),
                argMax(trd_buy_id, updated_at),
                argMax(lot_id, updated_at),
                argMax(customer_bin, updated_at),
                argMax(supplier_bin, updated_at),
                argMax(enstr_code_lvl4, updated_at),
                argMax(kato_code, updated_at),
                argMax(sign_date, updated_at),
                argMax(contract_sum, updated_at)
            FROM contracts
            WHERE id = %(contract_id)s
            GROUP BY id
            """,
            {"contract_id": contract_id},
        )
        if not rows:
            return 0

        row = rows[0]
        entity_id = _to_int(row[0])
        contract_number = _to_text(row[1])
        trd_buy_id = _to_int(row[2])
        lot_id = _to_int(row[3])
        customer_bin = _to_int(row[4])
        supplier_bin = _to_int(row[5])
        enstr_code = _to_text(row[6])
        kato_code = _to_text(row[7])
        sign_date = _to_iso_datetime(row[8])
        contract_sum = _to_float(row[9])

        content = (
            f"Contract #{contract_number}. Contract ID {entity_id}. Customer BIN {customer_bin}. "
            f"Supplier BIN {supplier_bin}. ENSTR code {enstr_code}. KATO {kato_code}. "
            f"Trade buy ID {trd_buy_id}. Lot ID {lot_id}. Signed on {sign_date}. "
            f"Contract sum {contract_sum:.2f}."
        )
        metadata = {
            "bin": customer_bin,
            "customer_bin": customer_bin,
            "supplier_bin": supplier_bin,
            "enstr_code": enstr_code,
            "kato_code": kato_code,
            "sign_date": sign_date,
            "contract_sum": contract_sum,
            "contract_number": contract_number,
            "trd_buy_id": trd_buy_id,
            "lot_id": lot_id,
        }
        return self._upsert_entity(
            entity_type="contract",
            procurement_id=entity_id,
            content=content,
            metadata=metadata,
        )

    def index_lot(self, lot_id: int, clickhouse_client: ClickHouseClient) -> int:
        rows = clickhouse_client.execute(
            """
            SELECT
                id,
                argMax(lot_number, updated_at),
                argMax(trd_buy_id, updated_at),
                argMax(customer_bin, updated_at),
                argMax(supplier_bin, updated_at),
                argMax(enstr_code_lvl4, updated_at),
                argMax(kato_code, updated_at),
                argMax(quantity, updated_at),
                argMax(amount, updated_at),
                argMax(created_at, updated_at)
            FROM lots
            WHERE id = %(lot_id)s
            GROUP BY id
            """,
            {"lot_id": lot_id},
        )
        if not rows:
            return 0

        row = rows[0]
        entity_id = _to_int(row[0])
        lot_number = _to_text(row[1])
        trd_buy_id = _to_int(row[2])
        customer_bin = _to_int(row[3])
        supplier_bin = _to_int(row[4])
        enstr_code = _to_text(row[5])
        kato_code = _to_text(row[6])
        quantity = _to_float(row[7])
        amount = _to_float(row[8])
        created_at = _to_iso_datetime(row[9])

        content = (
            f"Lot #{lot_number}. Lot ID {entity_id}. Announcement {trd_buy_id}. "
            f"Customer BIN {customer_bin}. Supplier BIN {supplier_bin}. ENSTR code {enstr_code}. "
            f"KATO {kato_code}. Quantity {quantity:.3f}. Amount {amount:.2f}. Created at {created_at}."
        )
        metadata = {
            "bin": customer_bin,
            "customer_bin": customer_bin,
            "supplier_bin": supplier_bin,
            "enstr_code": enstr_code,
            "kato_code": kato_code,
            "sign_date": created_at,
            "contract_sum": amount,
            "trd_buy_id": trd_buy_id,
            "lot_number": lot_number,
            "quantity": quantity,
        }
        return self._upsert_entity(
            entity_type="lot",
            procurement_id=entity_id,
            content=content,
            metadata=metadata,
        )

    def index_announcement(self, announcement_id: int, clickhouse_client: ClickHouseClient) -> int:
        rows = clickhouse_client.execute(
            """
            SELECT
                id,
                argMax(number_anno, updated_at),
                argMax(customer_bin, updated_at),
                argMax(total_sum, updated_at),
                argMax(count_lots, updated_at),
                argMax(ref_trade_methods_id, updated_at),
                argMax(ref_buy_status_id, updated_at),
                argMax(publish_date, updated_at),
                argMax(start_date, updated_at),
                argMax(end_date, updated_at)
            FROM announcements
            WHERE id = %(announcement_id)s
            GROUP BY id
            """,
            {"announcement_id": announcement_id},
        )
        if not rows:
            return 0

        row = rows[0]
        entity_id = _to_int(row[0])
        number_anno = _to_text(row[1])
        customer_bin = _to_int(row[2])
        total_sum = _to_float(row[3])
        count_lots = _to_int(row[4])
        trade_method_id = _to_int(row[5])
        status_id = _to_int(row[6])
        publish_date = _to_iso_datetime(row[7])
        start_date = _to_iso_datetime(row[8])
        end_date = _to_iso_datetime(row[9])

        content = (
            f"Announcement #{number_anno}. Announcement ID {entity_id}. Customer BIN {customer_bin}. "
            f"Total sum {total_sum:.2f}. Lots {count_lots}. Trade method {trade_method_id}. "
            f"Status {status_id}. Publish date {publish_date}. Start date {start_date}. End date {end_date}."
        )
        metadata = {
            "bin": customer_bin,
            "customer_bin": customer_bin,
            "supplier_bin": None,
            "enstr_code": "",
            "kato_code": "",
            "sign_date": publish_date,
            "contract_sum": total_sum,
            "number_anno": number_anno,
            "count_lots": count_lots,
            "trade_method_id": trade_method_id,
            "status_id": status_id,
            "start_date": start_date,
            "end_date": end_date,
        }
        return self._upsert_entity(
            entity_type="announcement",
            procurement_id=entity_id,
            content=content,
            metadata=metadata,
        )

    def index_plan(self, plan_id: int, clickhouse_client: ClickHouseClient) -> int:
        rows = clickhouse_client.execute(
            """
            SELECT
                id,
                argMax(plan_number, updated_at),
                argMax(customer_bin, updated_at),
                argMax(enstr_code_lvl4, updated_at),
                argMax(kato_code, updated_at),
                argMax(planned_amount, updated_at),
                argMax(quantity, updated_at),
                argMax(plan_year, updated_at),
                argMax(publish_date, updated_at)
            FROM plans
            WHERE id = %(plan_id)s
            GROUP BY id
            """,
            {"plan_id": plan_id},
        )
        if not rows:
            return 0

        row = rows[0]
        entity_id = _to_int(row[0])
        plan_number = _to_text(row[1])
        customer_bin = _to_int(row[2])
        enstr_code = _to_text(row[3])
        kato_code = _to_text(row[4])
        planned_amount = _to_float(row[5])
        quantity = _to_float(row[6])
        plan_year = _to_int(row[7])
        publish_date = _to_iso_datetime(row[8])

        content = (
            f"Plan #{plan_number}. Plan ID {entity_id}. Customer BIN {customer_bin}. "
            f"ENSTR code {enstr_code}. KATO {kato_code}. Planned amount {planned_amount:.2f}. "
            f"Quantity {quantity:.3f}. Plan year {plan_year}. Publish date {publish_date}."
        )
        metadata = {
            "bin": customer_bin,
            "customer_bin": customer_bin,
            "supplier_bin": None,
            "enstr_code": enstr_code,
            "kato_code": kato_code,
            "sign_date": publish_date,
            "contract_sum": planned_amount,
            "plan_number": plan_number,
            "quantity": quantity,
            "plan_year": plan_year,
        }
        return self._upsert_entity(
            entity_type="plan",
            procurement_id=entity_id,
            content=content,
            metadata=metadata,
        )

    def _build_filter(self, filters: Mapping[str, object] | None) -> object | None:
        if not filters:
            return None

        field_condition = _module_attr(self._models, "FieldCondition")
        match_any = _module_attr(self._models, "MatchAny")
        match_value = _module_attr(self._models, "MatchValue")
        match_text = _module_attr(self._models, "MatchText")
        datetime_range = _module_attr(self._models, "DatetimeRange")
        filter_factory = _module_attr(self._models, "Filter")

        conditions: list[object] = []

        customer_bin = filters.get("customer_bin")
        if isinstance(customer_bin, Sequence) and not isinstance(
            customer_bin, (str, bytes, bytearray)
        ):
            values = [_to_int(item) for item in customer_bin if _to_int(item) > 0]
            if values:
                conditions.append(
                    _build_instance(
                        field_condition,
                        key="customer_bin",
                        match=_build_instance(match_any, any=values),
                    )
                )
        elif customer_bin is not None:
            value = _to_int(customer_bin)
            if value > 0:
                conditions.append(
                    _build_instance(
                        field_condition,
                        key="customer_bin",
                        match=_build_instance(match_value, value=value),
                    )
                )

        enstr_code = _to_text(filters.get("enstr_code"))
        if enstr_code:
            conditions.append(
                _build_instance(
                    field_condition,
                    key="enstr_code",
                    match=_build_instance(match_text, text=enstr_code),
                )
            )

        date_from = _to_iso_datetime(filters.get("date_from"))
        date_to = _to_iso_datetime(filters.get("date_to"))
        if date_from or date_to:
            conditions.append(
                _build_instance(
                    field_condition,
                    key="sign_date",
                    range=_build_instance(
                        datetime_range,
                        gte=date_from if date_from else None,
                        lte=date_to if date_to else None,
                    ),
                )
            )

        if not conditions:
            return None
        return _build_instance(filter_factory, must=conditions)

    def search(
        self,
        query: str,
        filters: Mapping[str, object] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        query_vector = self.embed_text(query)
        query_filter = self._build_filter(filters)

        # Use query_points for qdrant-client v1.x (old .search() deprecated)
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results: list[SearchResult] = []
        for hit in response.points:
            payload_raw = hit.payload
            payload: Mapping[str, object]
            if isinstance(payload_raw, Mapping):
                payload = cast(Mapping[str, object], payload_raw)
            else:
                payload = cast(Mapping[str, object], {})
            metadata_raw = payload.get("metadata")
            metadata = (
                cast(dict[str, object], metadata_raw) if isinstance(metadata_raw, dict) else {}
            )
            for _field in (
                "customer_bin",
                "supplier_bin",
                "enstr_code",
                "kato_code",
                "sign_date",
                "contract_sum",
                "chunk_index",
                "total_chunks",
            ):
                if _field not in metadata and _field in payload:
                    metadata[_field] = payload[_field]
            results.append(
                SearchResult(
                    procurement_id=_to_int(payload.get("procurement_id"), default=0),
                    entity_type=_to_text(payload.get("entity_type")),
                    content=_to_text(payload.get("content")),
                    score=float(hit.score),
                    metadata=metadata,
                )
            )
        return results


def _index_sample(
    vector_store: VectorStore, clickhouse_client: ClickHouseClient, limit: int = 10
) -> int:
    rows = clickhouse_client.execute(
        """
        SELECT id
        FROM contracts
        GROUP BY id
        ORDER BY id DESC
        LIMIT %(limit)s
        """,
        {"limit": limit},
    )
    total_chunks = 0
    for row in rows:
        contract_id = _to_int(row[0], default=0)
        if contract_id <= 0:
            continue
        chunks = vector_store.index_contract(
            contract_id=contract_id, clickhouse_client=clickhouse_client
        )
        total_chunks += chunks
        LOGGER.info("indexed_contract id=%d chunks=%d", contract_id, chunks)
    return total_chunks


def main() -> None:
    try:
        from dotenv import load_dotenv

        _ = load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Qdrant vector store for procurement RAG")
    _index_sample_action = parser.add_argument(
        "--index-sample",
        action="store_true",
        help="Index contracts from ClickHouse",
    )
    _limit_action = parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of contracts to index (default: 10)",
    )
    _search_action = parser.add_argument(
        "--search", type=str, default="", help="Run semantic search"
    )
    _filters_action = parser.add_argument(
        "--filters",
        type=str,
        default="",
        help='JSON filters (customer_bin, enstr_code, date_from, date_to), e.g. {"customer_bin": [123]}',
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
    _collection_action = parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Collection name (default: procurement_documents)",
    )
    _ = (
        _index_sample_action,
        _search_action,
        _filters_action,
        _host_action,
        _port_action,
        _collection_action,
    )
    args = parser.parse_args()
    args_map = cast(dict[str, object], vars(args))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    qdrant_host = _to_text(
        args_map.get("qdrant_host") or os.getenv("QDRANT_HOST"),
        default=DEFAULT_QDRANT_HOST,
    )
    qdrant_port = _to_int(
        args_map.get("qdrant_port") or os.getenv("QDRANT_PORT"),
        default=DEFAULT_QDRANT_PORT,
    )
    vector_store = VectorStore(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=_to_text(args_map.get("collection"), default=DEFAULT_COLLECTION),
    )
    vector_store.ensure_collection()

    if bool(args_map.get("index_sample", False)):
        clickhouse = _get_clickhouse_client()
        try:
            limit = int(args_map.get("limit", 10))
            indexed = _index_sample(vector_store, clickhouse, limit=limit)
        finally:
            clickhouse.disconnect()
        LOGGER.info("sample_index_completed chunks=%d", indexed)

    search_query = _to_text(args_map.get("search"))
    if search_query:
        filters: dict[str, object] | None = None
        filters_raw = _to_text(args_map.get("filters"))
        if filters_raw:
            parsed_obj = cast(object, json.loads(filters_raw))
            if isinstance(parsed_obj, dict):
                filters = cast(dict[str, object], parsed_obj)

        for result in vector_store.search(search_query, filters=filters, limit=10):
            LOGGER.info(
                "search_result id=%d type=%s score=%.4f",
                result.procurement_id,
                result.entity_type,
                result.score,
            )


if __name__ == "__main__":
    main()
