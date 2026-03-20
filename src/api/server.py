from __future__ import annotations

# pyright: reportMissingImports=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUntypedFunctionDecorator=false, reportCallInDefaultInitializer=false, reportUnannotatedClassAttribute=false

import importlib
import json
import logging
import os
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.agent import generate_response
from src.analytics.queries import (
    spend_by_enstr,
    spend_by_region,
    supplier_concentration,
    total_spend_by_bin,
    year_over_year_trends,
)

LOGGER = logging.getLogger(__name__)

APP_VERSION = "1.0.0"
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000

ALLOWED_ANALYTICS_TYPES = {
    "spend_by_bin",
    "spend_by_enstr",
    "spend_by_region",
    "supplier_concentration",
    "yoy_trends",
}

ALLOWED_ENTITY_TYPES = {
    "subjects": {
        "table": "subjects",
        "date_field": "register_date",
        "search_fields": ("name_ru", "name_kz"),
    },
    "plans": {
        "table": "plans",
        "date_field": "publish_date",
        "search_fields": ("plan_number",),
    },
    "announcements": {
        "table": "announcements",
        "date_field": "publish_date",
        "search_fields": ("number_anno",),
    },
    "lots": {
        "table": "lots",
        "date_field": "created_at",
        "search_fields": ("lot_number",),
    },
    "contracts": {
        "table": "contracts",
        "date_field": "sign_date",
        "search_fields": ("contract_number",),
    },
}


class ClickHouseClient(Protocol):
    def execute(self, query: str, params: object | None = None) -> list[tuple[object, ...]]: ...

    def disconnect(self) -> None: ...


class HealthResponse(BaseModel):
    status: Literal["ok"]
    timestamp: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    language: str = Field(default="auto", pattern=r"^(auto|ru|kz|russian|kazakh)$")


class QueryResponse(BaseModel):
    answer: str
    metadata: dict[str, Any]


class AnomalyFilter(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    anomaly_type: str | None = Field(default=None, alias="type")
    customer_bin: str | None = Field(default=None, alias="bin")
    date_from: date | None = None
    date_to: date | None = None


class AnomalyItem(BaseModel):
    id: int
    anomaly_type: str
    entity_type: str
    entity_id: int
    severity: str
    detected_at: str
    deviation_pct: str
    expected_value: str
    actual_value: str
    sample_size: int
    enstr_code: str
    kato_code: str | None
    metadata: dict[str, Any] | str


class AnomaliesResponse(BaseModel):
    anomalies: list[AnomalyItem]
    total: int
    filters: dict[str, Any]
    limit: int
    offset: int


class AnalyticsRequest(BaseModel):
    bins: list[str] | None = None
    enstr_codes: list[str] | None = None
    kato_codes: list[str] | None = None
    date_from: date | None = None
    date_to: date | None = None


class AnalyticsResponse(BaseModel):
    analytics_type: str
    data: list[dict[str, Any]]
    filters: dict[str, Any]
    generated_at: str


class EntitiesResponse(BaseModel):
    entity_type: str
    data: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


_FALLBACK_ENCODINGS = ("cp1251", "cp866", "latin-1")


class _UTF8NormalizationMiddleware:
    """Re-encodes non-UTF-8 request bodies (e.g. cp1251 from Windows CMD) to UTF-8."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        if method in ("GET", "HEAD", "OPTIONS", "DELETE"):
            await self._app(scope, receive, send)
            return

        body_chunks: list[bytes] = []
        body_complete = False

        async def _buffered_receive() -> Message:
            nonlocal body_complete

            if body_complete:
                # Empty sentinel — prevents Starlette from hanging after body is consumed.
                return {"type": "http.request", "body": b"", "more_body": False}

            message = await receive()
            if message["type"] == "http.request":
                body_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    body_complete = True
                    raw_body = b"".join(body_chunks)
                    normalized = self._normalize(raw_body)
                    return {"type": "http.request", "body": normalized, "more_body": False}
            return message

        await self._app(scope, _buffered_receive, send)

    @staticmethod
    def _normalize(raw: bytes) -> bytes:
        if not raw:
            return raw
        try:
            raw.decode("utf-8")
            return raw
        except UnicodeDecodeError:
            pass

        for encoding in _FALLBACK_ENCODINGS:
            try:
                decoded = raw.decode(encoding)
                LOGGER.info("request_body_reencoded from=%s bytes=%d", encoding, len(raw))
                return decoded.encode("utf-8")
            except (UnicodeDecodeError, LookupError):
                continue

        LOGGER.warning("request_body_encoding_unknown bytes=%d", len(raw))
        return raw


app = FastAPI(
    title="Kazakhstan Procurement AI Agent API",
    description="AI-powered analysis of Kazakhstan public procurement data",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(_UTF8NormalizationMiddleware)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Any) -> JSONResponse:
    start_ts = datetime.now(UTC)
    try:
        response = await call_next(request)
        duration_ms = (datetime.now(UTC) - start_ts).total_seconds() * 1000
        LOGGER.info(
            "request_completed method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
    except Exception:
        duration_ms = (datetime.now(UTC) - start_ts).total_seconds() * 1000
        LOGGER.exception(
            "request_failed method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_yaml(path: Path) -> Any:
    yaml_module = importlib.import_module("yaml")
    safe_load = cast(Any, getattr(yaml_module, "safe_load"))
    return safe_load(path.read_text(encoding="utf-8"))


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    loaded = _load_yaml(config_path)
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml must be a dictionary at top level")
    return cast(dict[str, Any], loaded)


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return default
    try:
        return int(digits)
    except ValueError:
        return default


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        normalized = value.astimezone(UTC) if value.tzinfo is not None else value
        return normalized.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _build_clickhouse_client(config: dict[str, Any]) -> ClickHouseClient:
    clickhouse_cfg = (
        cast(dict[str, Any], config.get("database", {})).get("clickhouse", {})
        if isinstance(config.get("database"), dict)
        else {}
    )
    host = os.getenv("CLICKHOUSE_HOST") or str(clickhouse_cfg.get("host", "")).strip()
    if not host:
        raise ValueError("ClickHouse host is not configured")

    port = _to_int(os.getenv("CLICKHOUSE_PORT") or clickhouse_cfg.get("port") or 9440, 9440)
    database = os.getenv("CLICKHOUSE_DB") or str(clickhouse_cfg.get("database", "")).strip()
    if not database:
        raise ValueError("ClickHouse database is not configured")
    user = os.getenv("CLICKHOUSE_USER", "default")
    password = os.getenv("CLICKHOUSE_PASSWORD", "")

    is_secure = port in (9440, 8443)
    clickhouse_driver = importlib.import_module("clickhouse_driver")
    client_factory = cast(Any, getattr(clickhouse_driver, "Client"))
    return cast(
        ClickHouseClient,
        client_factory(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            secure=is_secure,
            connect_timeout=10,
            send_receive_timeout=10,
        ),
    )


def _execute_clickhouse(
    config: dict[str, Any], query: str, params: dict[str, Any]
) -> list[tuple[Any, ...]]:
    client = _build_clickhouse_client(config)
    try:
        return cast(list[tuple[Any, ...]], client.execute(query, params))
    finally:
        client.disconnect()


def _parse_csv_or_multi(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    expanded: list[str] = []
    for item in values:
        for part in str(item).split(","):
            candidate = part.strip()
            if candidate:
                expanded.append(candidate)
    return expanded or None


def _parse_metadata_json(raw: Any) -> dict[str, Any] | str:
    if isinstance(raw, dict):
        return cast(dict[str, Any], _jsonable(raw))
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
        return cast(dict[str, Any], loaded) if isinstance(loaded, dict) else text
    except json.JSONDecodeError:
        return text


def _analytics_rows_to_dicts(
    analytics_type: str, rows: list[tuple[Any, ...]]
) -> list[dict[str, Any]]:
    if analytics_type == "spend_by_bin":
        return [
            {
                "bin": str(row[0]),
                "total_spend": _jsonable(row[1]),
                "contract_count": _jsonable(row[2]),
            }
            for row in rows
            if len(row) >= 3
        ]
    if analytics_type == "spend_by_enstr":
        return [
            {
                "enstr_code": _jsonable(row[0]),
                "total_spend": _jsonable(row[1]),
                "contract_count": _jsonable(row[2]),
            }
            for row in rows
            if len(row) >= 3
        ]
    if analytics_type == "spend_by_region":
        return [
            {
                "kato_code": _jsonable(row[0]),
                "region_name": _jsonable(row[1]),
                "total_spend": _jsonable(row[2]),
                "contract_count": _jsonable(row[3]),
            }
            for row in rows
            if len(row) >= 4
        ]
    if analytics_type == "supplier_concentration":
        return [
            {
                "customer_bin": _jsonable(row[0]),
                "hhi_score": _jsonable(row[1]),
                "supplier_count": _jsonable(row[2]),
                "top_supplier_share": _jsonable(row[3]),
            }
            for row in rows
            if len(row) >= 4
        ]
    if analytics_type == "yoy_trends":
        return [
            {
                "year": _jsonable(row[0]),
                "total_spend": _jsonable(row[1]),
                "contract_count": _jsonable(row[2]),
                "yoy_change_pct": _jsonable(row[3]),
            }
            for row in rows
            if len(row) >= 4
        ]
    return []


def _date_to_datetime_bounds(day: date, end_of_day: bool = False) -> datetime:
    if end_of_day:
        return datetime(day.year, day.month, day.day, 23, 59, 59)
    return datetime(day.year, day.month, day.day, 0, 0, 0)


async def _startup_validate() -> None:
    config_path = Path(os.getenv("APP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
    config = _load_config(config_path)
    app.state.config = config
    app.state.config_path = config_path
    app.state.startup_checks = {
        "config": True,
        "clickhouse": False,
        "qdrant": False,
        "checked_at": _utcnow_iso(),
    }

    def _check_clickhouse() -> bool:
        try:
            client = _build_clickhouse_client(config)
            try:
                _ = client.execute("SELECT 1")
                return True
            finally:
                client.disconnect()
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("startup_clickhouse_check_failed error=%s", type(error).__name__)
            return False

    def _check_qdrant() -> bool:
        try:
            qdrant_host = str(
                os.getenv("QDRANT_HOST")
                or cast(dict[str, Any], config.get("qdrant", {})).get("host", "localhost")
            ).strip()
            qdrant_port = _to_int(
                os.getenv("QDRANT_PORT")
                or cast(dict[str, Any], config.get("qdrant", {})).get("port", 6333),
                6333,
            )
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            qdrant_module = importlib.import_module("qdrant_client")
            qdrant_client_factory = cast(Any, getattr(qdrant_module, "QdrantClient"))

            if qdrant_api_key:
                client = qdrant_client_factory(url=qdrant_host, api_key=qdrant_api_key)
            elif qdrant_host.startswith("http://") or qdrant_host.startswith("https://"):
                client = qdrant_client_factory(url=qdrant_host)
            else:
                client = qdrant_client_factory(host=qdrant_host, port=qdrant_port)

            _ = client.get_collections()
            return True
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("startup_qdrant_check_failed error=%s", type(error).__name__)
            return False

    app.state.startup_checks["clickhouse"] = await run_in_threadpool(_check_clickhouse)
    app.state.startup_checks["qdrant"] = await run_in_threadpool(_check_qdrant)


@app.on_event("startup")
async def startup_event() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    message = str(exc.detail) if exc.detail else "Request failed"
    payload = ErrorResponse(error="http_error", detail=message, timestamp=_utcnow_iso())
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    payload = ErrorResponse(
        error="validation_error",
        detail="Request validation failed",
        timestamp=_utcnow_iso(),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("unhandled_exception path=%s", request.url.path)
    payload = ErrorResponse(
        error="internal_server_error",
        detail="Internal server error",
        timestamp=_utcnow_iso(),
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


def _config_from_request(request: Request) -> dict[str, Any]:
    config = cast(dict[str, Any] | None, getattr(request.app.state, "config", None))
    if config is None:
        config_path = Path(os.getenv("APP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
        config = _load_config(config_path)
        request.app.state.config = config
        request.app.state.config_path = config_path
    return config


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    responses={500: {"model": ErrorResponse}},
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=_utcnow_iso(), version=APP_VERSION)


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["query"],
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def query_endpoint(payload: QueryRequest, request: Request) -> QueryResponse:
    config_path = cast(Path | None, getattr(request.app.state, "config_path", None))
    if config_path is None:
        config_path = Path(os.getenv("APP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
        request.app.state.config_path = config_path

    generated = await run_in_threadpool(generate_response, payload.question, config_path)
    sections = cast(dict[str, Any], generated.get("sections", {}))
    answer = str(generated.get("answer", sections.get("summary", "")))
    limits = sections.get("limitations_and_confidence") or sections.get("limitations_confidence")
    if not isinstance(limits, dict):
        limits = {}

    metadata: dict[str, Any] = {
        "data_used": cast(dict[str, Any], _jsonable(sections.get("data_used", {}))),
        "comparison": cast(dict[str, Any], _jsonable(sections.get("comparison", {}))),
        "evaluation_metric": cast(dict[str, Any], _jsonable(sections.get("evaluation_metric", {}))),
        "limitations_confidence": cast(dict[str, Any], _jsonable(limits)),
        "examples": cast(list[dict[str, Any]], _jsonable(sections.get("examples", []))),
        "links": cast(list[str], _jsonable(sections.get("links", []))),
    }

    return QueryResponse(
        answer=answer,
        metadata=metadata,
    )


def _build_anomaly_filter(
    anomaly_type: str | None = Query(default=None, alias="type"),
    customer_bin: str | None = Query(default=None, alias="bin"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
) -> AnomalyFilter:
    return AnomalyFilter(
        type=anomaly_type,
        bin=customer_bin,
        date_from=date_from,
        date_to=date_to,
    )


@app.get(
    "/anomalies",
    response_model=AnomaliesResponse,
    tags=["anomalies"],
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def anomalies_endpoint(
    request: Request,
    filters: AnomalyFilter = Depends(_build_anomaly_filter),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> AnomaliesResponse:
    config = _config_from_request(request)
    where_parts = ["1 = 1"]
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    if filters.anomaly_type:
        where_parts.append("anomaly_type = %(anomaly_type)s")
        params["anomaly_type"] = filters.anomaly_type
    if filters.customer_bin:
        where_parts.append("positionCaseInsensitiveUTF8(metadata, %(customer_bin)s) > 0")
        params["customer_bin"] = filters.customer_bin
    if filters.date_from:
        where_parts.append("detected_at >= %(date_from)s")
        params["date_from"] = _date_to_datetime_bounds(filters.date_from)
    if filters.date_to:
        where_parts.append("detected_at <= %(date_to)s")
        params["date_to"] = _date_to_datetime_bounds(filters.date_to, end_of_day=True)

    where_clause = " AND ".join(where_parts)

    list_sql = f"""
    SELECT
        id,
        anomaly_type,
        entity_type,
        entity_id,
        severity,
        detected_at,
        deviation_pct,
        expected_value,
        actual_value,
        sample_size,
        enstr_code,
        kato_code,
        metadata
    FROM anomaly_results
    WHERE {where_clause}
    ORDER BY detected_at DESC, id DESC
    LIMIT %(limit)s OFFSET %(offset)s
    """
    count_sql = f"SELECT count() FROM anomaly_results WHERE {where_clause}"

    rows = await run_in_threadpool(_execute_clickhouse, config, list_sql, params)
    count_rows = await run_in_threadpool(_execute_clickhouse, config, count_sql, params)
    total = _to_int(count_rows[0][0] if count_rows else 0, 0)

    anomalies: list[AnomalyItem] = []
    for row in rows:
        if len(row) < 13:
            continue
        anomalies.append(
            AnomalyItem(
                id=_to_int(row[0]),
                anomaly_type=str(row[1]),
                entity_type=str(row[2]),
                entity_id=_to_int(row[3]),
                severity=str(row[4]),
                detected_at=str(_jsonable(row[5])),
                deviation_pct=str(_jsonable(row[6])),
                expected_value=str(_jsonable(row[7])),
                actual_value=str(_jsonable(row[8])),
                sample_size=_to_int(row[9]),
                enstr_code=str(row[10]) if row[10] is not None else "",
                kato_code=str(row[11]) if row[11] is not None else None,
                metadata=_parse_metadata_json(row[12]),
            )
        )

    return AnomaliesResponse(
        anomalies=anomalies,
        total=total,
        filters=filters.model_dump(exclude_none=True, by_alias=True),
        limit=limit,
        offset=offset,
    )


@app.get(
    "/analytics/{analytics_type}",
    response_model=AnalyticsResponse,
    tags=["analytics"],
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analytics_endpoint(
    analytics_type: str,
    request: Request,
    bins: list[str] | None = Query(default=None),
    enstr_codes: list[str] | None = Query(default=None),
    kato_codes: list[str] | None = Query(default=None),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
) -> AnalyticsResponse:
    if analytics_type not in ALLOWED_ANALYTICS_TYPES:
        raise HTTPException(status_code=404, detail=f"Unknown analytics type: {analytics_type}")

    config = _config_from_request(request)
    filters = AnalyticsRequest(
        bins=_parse_csv_or_multi(bins),
        enstr_codes=_parse_csv_or_multi(enstr_codes),
        kato_codes=_parse_csv_or_multi(kato_codes),
        date_from=date_from,
        date_to=date_to,
    )

    parsed_bins = [_to_int(item) for item in filters.bins or [] if _to_int(item) > 0]
    parsed_enstr = [str(item) for item in (filters.enstr_codes or [])]
    parsed_kato = [str(item) for item in (filters.kato_codes or [])]
    start_dt = _date_to_datetime_bounds(filters.date_from) if filters.date_from else None
    end_dt = _date_to_datetime_bounds(filters.date_to, end_of_day=True) if filters.date_to else None

    def _run_analytics() -> list[tuple[Any, ...]]:
        client = _build_clickhouse_client(config)
        try:
            if analytics_type == "spend_by_bin":
                return cast(
                    list[tuple[Any, ...]],
                    total_spend_by_bin(
                        client,
                        bins=parsed_bins or None,
                        start_date=start_dt,
                        end_date=end_dt,
                        enstr_codes=parsed_enstr or None,
                        kato_codes=parsed_kato or None,
                    ),
                )
            if analytics_type == "spend_by_enstr":
                return cast(
                    list[tuple[Any, ...]],
                    spend_by_enstr(
                        client,
                        bins=parsed_bins or None,
                        start_date=start_dt,
                        end_date=end_dt,
                        enstr_codes=parsed_enstr or None,
                        kato_codes=parsed_kato or None,
                    ),
                )
            if analytics_type == "spend_by_region":
                return cast(
                    list[tuple[Any, ...]],
                    spend_by_region(
                        client,
                        bins=parsed_bins or None,
                        start_date=start_dt,
                        end_date=end_dt,
                        enstr_codes=parsed_enstr or None,
                        kato_codes=parsed_kato or None,
                    ),
                )
            if analytics_type == "supplier_concentration":
                return cast(
                    list[tuple[Any, ...]],
                    supplier_concentration(
                        client,
                        bins=parsed_bins or None,
                        start_date=start_dt,
                        end_date=end_dt,
                        enstr_codes=parsed_enstr or None,
                        kato_codes=parsed_kato or None,
                    ),
                )
            return cast(
                list[tuple[Any, ...]],
                year_over_year_trends(
                    client,
                    bins=parsed_bins or None,
                    start_date=start_dt,
                    end_date=end_dt,
                    enstr_codes=parsed_enstr or None,
                    kato_codes=parsed_kato or None,
                ),
            )
        finally:
            client.disconnect()

    raw_rows = await run_in_threadpool(_run_analytics)
    return AnalyticsResponse(
        analytics_type=analytics_type,
        data=_analytics_rows_to_dicts(analytics_type, raw_rows),
        filters=filters.model_dump(exclude_none=True),
        generated_at=_utcnow_iso(),
    )


@app.get(
    "/entities/{entity_type}",
    response_model=EntitiesResponse,
    tags=["entities"],
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def entities_endpoint(
    entity_type: str,
    request: Request,
    q: str | None = Query(default=None, min_length=1, max_length=200),
    customer_bin: str | None = Query(default=None),
    supplier_bin: str | None = Query(default=None),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> EntitiesResponse:
    if entity_type not in ALLOWED_ENTITY_TYPES:
        raise HTTPException(status_code=404, detail=f"Unknown entity type: {entity_type}")

    config = _config_from_request(request)
    entity_cfg = cast(dict[str, Any], ALLOWED_ENTITY_TYPES[entity_type])
    table = str(entity_cfg["table"])
    date_field = str(entity_cfg["date_field"])
    search_fields = cast(tuple[str, ...], entity_cfg["search_fields"])

    where_parts = ["1 = 1"]
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    if customer_bin:
        where_parts.append("toString(customer_bin) = %(customer_bin)s")
        params["customer_bin"] = customer_bin
    if supplier_bin and table in {"lots", "contracts"}:
        where_parts.append("toString(supplier_bin) = %(supplier_bin)s")
        params["supplier_bin"] = supplier_bin
    if date_from:
        where_parts.append(f"{date_field} >= %(date_from)s")
        params["date_from"] = _date_to_datetime_bounds(date_from)
    if date_to:
        where_parts.append(f"{date_field} <= %(date_to)s")
        params["date_to"] = _date_to_datetime_bounds(date_to, end_of_day=True)
    if q:
        search_conds = [
            f"positionCaseInsensitiveUTF8(toString({field}), %(q)s) > 0" for field in search_fields
        ]
        where_parts.append(f"({' OR '.join(search_conds)})")
        params["q"] = q

    where_clause = " AND ".join(where_parts)

    describe_sql = f"DESCRIBE TABLE {table}"
    describe_rows = await run_in_threadpool(_execute_clickhouse, config, describe_sql, {})
    column_names = [str(row[0]) for row in describe_rows if len(row) > 0]
    if not column_names or "id" not in column_names:
        raise HTTPException(status_code=500, detail=f"Unable to inspect table schema: {table}")

    argmax_columns: list[str] = []
    for column in column_names:
        if column == "id":
            continue
        argmax_columns.append(f"argMax({column}, updated_at) AS {column}")

    latest_projection = "\n            ".join(["id"] + argmax_columns)
    latest_sql = f"""
    WITH latest AS (
        SELECT
            {latest_projection}
        FROM {table}
        GROUP BY id
    )
    SELECT *
    FROM latest
    WHERE {where_clause}
    ORDER BY id DESC
    LIMIT %(limit)s OFFSET %(offset)s
    """
    count_sql = f"""
    WITH latest AS (
        SELECT
            {latest_projection}
        FROM {table}
        GROUP BY id
    )
    SELECT count()
    FROM latest
    WHERE {where_clause}
    """

    rows = await run_in_threadpool(_execute_clickhouse, config, latest_sql, params)
    count_rows = await run_in_threadpool(_execute_clickhouse, config, count_sql, params)
    total = _to_int(count_rows[0][0] if count_rows else 0)

    data: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for idx, column in enumerate(column_names):
            value = row[idx] if idx < len(row) else None
            item[column] = _jsonable(value)
        data.append(item)

    return EntitiesResponse(
        entity_type=entity_type,
        data=data,
        total=total,
        limit=limit,
        offset=offset,
    )
