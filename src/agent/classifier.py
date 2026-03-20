from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, Protocol, TypedDict, cast

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH: Final[Path] = Path("config.yaml")
DEFAULT_TIMEOUT_SECONDS: Final[int] = 30

QUERY_TYPES: Final[tuple[str, ...]] = (
    "SEARCH",
    "COMPARISON",
    "ANALYTICS",
    "ANOMALY_DETECTION",
    "FAIRNESS",
)
LANGUAGES: Final[tuple[str, ...]] = ("kazakh", "russian", "unknown")

QueryType = Literal[
    "SEARCH",
    "COMPARISON",
    "ANALYTICS",
    "ANOMALY_DETECTION",
    "FAIRNESS",
]
Language = Literal["kazakh", "russian", "unknown"]
Method = Literal["llm", "pattern", "fallback"]

QUERY_TYPE_ALIASES: Final[dict[str, QueryType]] = {
    "SEARCH": "SEARCH",
    "COMPARISON": "COMPARISON",
    "ANALYTICS": "ANALYTICS",
    "ANOMALY_DETECTION": "ANOMALY_DETECTION",
    "ANOMALY": "ANOMALY_DETECTION",
    "FAIRNESS": "FAIRNESS",
}
LANGUAGE_ALIASES: Final[dict[str, Language]] = {
    "kazakh": "kazakh",
    "russian": "russian",
    "unknown": "unknown",
    "kk": "kazakh",
    "kz": "kazakh",
    "ru": "russian",
    "rus": "russian",
}

KAZAKH_SPECIFIC_CHARS: Final[set[str]] = set("әғқңөұүһі")
RUSSIAN_SPECIFIC_CHARS: Final[set[str]] = set("ёъыэюя")
KAZAKH_HINT_TOKENS: Final[set[str]] = {
    "қандай",
    "қалай",
    "баға",
    "сатып",
    "алулар",
    "салыстыру",
    "әділ",
    "бар",
}
RUSSIAN_HINT_TOKENS: Final[set[str]] = {
    "какие",
    "какой",
    "цена",
    "закупки",
    "сравнить",
    "справедливость",
    "покажи",
    "найти",
}

BIN_REGEX: Final[re.Pattern[str]] = re.compile(r"\b\d{12}\b")
DATE_YYYY_MM_DD_REGEX: Final[re.Pattern[str]] = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DATE_DD_MM_YYYY_REGEX: Final[re.Pattern[str]] = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
DATE_YYYY_MM_REGEX: Final[re.Pattern[str]] = re.compile(r"\b\d{4}-(?:0[1-9]|1[0-2])\b(?!-\d{2}\b)")
DATE_YEAR_REGEX: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(2024|2025|2026)(?:(?:\s*г(?:\.|ода)?)|(?:-?го))?(?!\d)(?!-\d{2}\b)",
    flags=re.IGNORECASE,
)
_MONTH_NAMES_PATTERN: Final[str] = (
    # Russian genitive: "января 2025", "февраля 2025", ...
    r"января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|"
    # Russian prepositional: "в январе 2025", "в феврале 2025", ...
    r"январе|феврале|марте|апреле|мае|июне|июле|августе|сентябре|октябре|ноябре|декабре|"
    # Russian accusative: "за январь 2025", "за февраль 2025", ...
    r"январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь|"
    # Kazakh month names (nominative)
    r"қаңтар|ақпан|наурыз|сәуір|мамыр|маусым|шілде|тамыз|қыркүйек|қазан|қараша|желтоқсан"
)
DATE_MONTH_YEAR_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"\b({_MONTH_NAMES_PATTERN})\s+(2024|2025|2026)\b",
    flags=re.IGNORECASE,
)
# "январь и февраль 2025", "март, апрель 2025" — first month shares the year with the second
DATE_MULTI_MONTH_YEAR_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"\b({_MONTH_NAMES_PATTERN})(?:\s*[,и]\s+)({_MONTH_NAMES_PATTERN})\s+(2024|2025|2026)\b",
    flags=re.IGNORECASE,
)
ENSTR_REGEX: Final[re.Pattern[str]] = re.compile(r"\b\d{2}(?:\.\d{2,4}){1,3}\b")

_DATE_GRANULARITY = Literal["day", "month", "year"]

_MONTH_TOKENS: Final[dict[str, int]] = {
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

KEYWORDS_BY_TYPE: Final[dict[QueryType, tuple[str, ...]]] = {
    "SEARCH": (
        "найти",
        "табу",
        "искать",
        "іздеу",
        "search",
        "contract",
        "контракт",
        "келісімшарт",
        "лот",
        "поставщик",
        "жеткізуші",
    ),
    "COMPARISON": (
        "сравнить",
        "салыстыру",
        "салыстыр",
        "difference",
        "айырмашылық",
        "разница",
        "compare",
        "vs",
    ),
    "ANALYTICS": (
        "статистика",
        "статистик",
        "статистикасы",
        "аналитика",
        "талдау",
        "aggregate",
        "общий",
        "жалпы",
        "закуп",
        "trend",
        "динамика",
    ),
    "ANOMALY_DETECTION": (
        "anomaly",
        "аномалия",
        "аномалии",
        "күдікті",
        "подозр",
        "suspicious",
        "отклонение",
    ),
    "FAIRNESS": (
        "fairness",
        "әділ",
        "справедлив",
        "reasonable price",
        "орынды баға",
        "обоснован",
        "бағала",
    ),
}


class Entities(TypedDict):
    bins: list[str]
    dates: list[str]
    enstr_codes: list[str]


class ClassificationResult(TypedDict):
    query_type: QueryType
    language: Language
    entities: Entities
    confidence: float
    method: Method


class _PartialClassification(TypedDict, total=False):
    query_type: QueryType
    language: Language
    entities: Entities
    confidence: float


class _YamlSafeLoad(Protocol):
    def __call__(self, stream: str) -> object: ...


class _LLMClientProtocol(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: object,
    ) -> str: ...


def _module_attr(module: ModuleType, name: str) -> object:
    return cast(object, getattr(module, name))


def _build_instance(factory: object, *args: object, **kwargs: object) -> object:
    callable_factory = cast(Callable[..., object], factory)
    return callable_factory(*args, **kwargs)


def _load_yaml(path: Path) -> object:
    yaml_module = importlib.import_module("yaml")
    safe_load = cast(_YamlSafeLoad, _module_attr(yaml_module, "safe_load"))
    return safe_load(path.read_text(encoding="utf-8"))


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return None


def _normalized_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in values:
        value = raw.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _empty_entities() -> Entities:
    return {"bins": [], "dates": [], "enstr_codes": []}


def _merge_entities(*groups: Entities) -> Entities:
    bins: list[str] = []
    dates: list[str] = []
    enstr_codes: list[str] = []
    for group in groups:
        bins.extend(group.get("bins", []))
        dates.extend(group.get("dates", []))
        enstr_codes.extend(group.get("enstr_codes", []))
    return {
        "bins": _normalized_unique(bins),
        "dates": _drop_subsumed_year_dates(_normalized_unique(dates)),
        "enstr_codes": _normalized_unique(enstr_codes),
    }


def _parse_date_token_granularity(value: str) -> tuple[_DATE_GRANULARITY, int] | None:
    normalized = value.strip()
    if not normalized:
        return None

    if re.fullmatch(r"\d{4}", normalized):
        return ("year", int(normalized))

    if re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])-\d{2}", normalized):
        return ("day", int(normalized[:4]))

    if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", normalized):
        return ("day", int(normalized[-4:]))

    if re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", normalized):
        return ("month", int(normalized[:4]))

    month_year_match = re.fullmatch(r"([\wәғқңөұүһіё]+)\s+(\d{4})", normalized.lower())
    if month_year_match and month_year_match.group(1) in _MONTH_TOKENS:
        return ("month", int(month_year_match.group(2)))
    return None


def _drop_subsumed_year_dates(values: Sequence[str]) -> list[str]:
    parsed = [_parse_date_token_granularity(value) for value in values]
    years_with_month_or_day = {
        year
        for item in parsed
        if item is not None
        for granularity, year in [item]
        if granularity in {"month", "day"}
    }
    output: list[str] = []
    for value, parsed_item in zip(values, parsed, strict=False):
        if parsed_item is None:
            output.append(value)
            continue
        granularity, year = parsed_item
        if granularity == "year" and year in years_with_month_or_day:
            continue
        output.append(value)
    return output


def _to_query_type(value: object) -> QueryType | None:
    text = str(value).strip().upper()
    return QUERY_TYPE_ALIASES.get(text)


def _to_language(value: object) -> Language | None:
    text = str(value).strip().lower()
    return LANGUAGE_ALIASES.get(text)


def _to_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _clamp_confidence(value: object, *, default: float) -> float:
    return max(0.0, min(1.0, _to_float(value, default=default)))


def _to_entities(value: object) -> Entities:
    if not isinstance(value, Mapping):
        return _empty_entities()
    typed_value = cast(Mapping[str, object], value)

    bins_raw = typed_value.get("bins", [])
    dates_raw = typed_value.get("dates", [])
    enstr_raw = typed_value.get("enstr_codes", [])

    bins = (
        [str(item).strip() for item in bins_raw if str(item).strip()]
        if isinstance(bins_raw, Sequence) and not isinstance(bins_raw, (str, bytes, bytearray))
        else []
    )
    dates = (
        [str(item).strip() for item in dates_raw if str(item).strip()]
        if isinstance(dates_raw, Sequence) and not isinstance(dates_raw, (str, bytes, bytearray))
        else []
    )
    enstr_codes = (
        [str(item).strip() for item in enstr_raw if str(item).strip()]
        if isinstance(enstr_raw, Sequence) and not isinstance(enstr_raw, (str, bytes, bytearray))
        else []
    )

    return {
        "bins": _normalized_unique(bins),
        "dates": _normalized_unique(dates),
        "enstr_codes": _normalized_unique(enstr_codes),
    }


def _extract_json_object(payload: str) -> str:
    stripped = payload.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Response does not contain a JSON object")
    return stripped[start : end + 1]


def detect_language(query: str) -> Language:
    lowered = query.lower()
    kazakh_char_hits = sum(1 for ch in lowered if ch in KAZAKH_SPECIFIC_CHARS)
    russian_char_hits = sum(1 for ch in lowered if ch in RUSSIAN_SPECIFIC_CHARS)
    has_kazakh = kazakh_char_hits > 0
    has_russian = russian_char_hits > 0

    if has_kazakh and not has_russian:
        return "kazakh"
    if has_russian and not has_kazakh:
        return "russian"

    tokens = {
        token
        for token in cast(
            list[str], re.findall(r"[а-яәғқңөұүһіёъыэюя]+", lowered, flags=re.IGNORECASE)
        )
        if token
    }
    kazakh_hint_hits = len(tokens & KAZAKH_HINT_TOKENS)
    russian_hint_hits = len(tokens & RUSSIAN_HINT_TOKENS)

    if has_kazakh and has_russian:
        if kazakh_hint_hits > russian_hint_hits:
            return "kazakh"
        if russian_hint_hits > kazakh_hint_hits:
            return "russian"
        if kazakh_char_hits > russian_char_hits:
            return "kazakh"
        if russian_char_hits > kazakh_char_hits:
            return "russian"

    if kazakh_hint_hits > russian_hint_hits:
        return "kazakh"
    if russian_hint_hits > kazakh_hint_hits:
        return "russian"
    return "unknown"


def _extract_bins(query: str, configured_bins: set[str]) -> list[str]:
    matches = cast(list[str], BIN_REGEX.findall(query))
    if not matches:
        return []
    preferred = [value for value in matches if value in configured_bins]
    if preferred:
        return _normalized_unique(preferred)
    return _normalized_unique(matches)


def _extract_dates(query: str) -> list[str]:
    values: list[str] = []
    yyyy_mm_dd_matches = cast(list[str], DATE_YYYY_MM_DD_REGEX.findall(query))
    dd_mm_yyyy_matches = cast(list[str], DATE_DD_MM_YYYY_REGEX.findall(query))
    yyyy_mm_matches = cast(list[str], DATE_YYYY_MM_REGEX.findall(query))

    values.extend(yyyy_mm_dd_matches)
    values.extend(dd_mm_yyyy_matches)
    values.extend(yyyy_mm_matches)

    month_year_matches = cast(list[tuple[str, str]], DATE_MONTH_YEAR_REGEX.findall(query))
    month_year_tokens = [f"{month} {year}" for month, year in month_year_matches]

    multi_month_matches = cast(
        list[tuple[str, str, str]], DATE_MULTI_MONTH_YEAR_REGEX.findall(query)
    )
    for first_month, _second_month, year in multi_month_matches:
        token = f"{first_month} {year}"
        if token not in month_year_tokens:
            month_year_tokens.append(token)

    subsumed_years = {year for _, year in month_year_matches}
    subsumed_years.update(year for _, _, year in multi_month_matches)
    subsumed_years.update(token[:4] for token in yyyy_mm_dd_matches)
    subsumed_years.update(token[-4:] for token in dd_mm_yyyy_matches)
    subsumed_years.update(token[:4] for token in yyyy_mm_matches)

    year_matches = cast(list[str], DATE_YEAR_REGEX.findall(query))
    values.extend(year for year in year_matches if year not in subsumed_years)
    values.extend(month_year_tokens)
    return _drop_subsumed_year_dates(_normalized_unique(values))


def _extract_enstr_codes(query: str) -> list[str]:
    return _normalized_unique(cast(list[str], ENSTR_REGEX.findall(query)))


def extract_entities(query: str, configured_bins: set[str]) -> Entities:
    return {
        "bins": _extract_bins(query, configured_bins),
        "dates": _extract_dates(query),
        "enstr_codes": _extract_enstr_codes(query),
    }


def _keyword_score(query_lower: str, keywords: Sequence[str]) -> int:
    return sum(1 for keyword in keywords if keyword in query_lower)


def _pattern_query_type(query: str) -> tuple[QueryType, float, Method]:
    query_lower = query.lower()
    scores: dict[QueryType, int] = {
        qtype: _keyword_score(query_lower, keywords) for qtype, keywords in KEYWORDS_BY_TYPE.items()
    }

    best_type = max(scores, key=lambda qtype: scores[qtype])
    best_score = scores[best_type]
    ranked = sorted(scores.values(), reverse=True)
    runner_up = ranked[1] if len(ranked) > 1 else 0

    if best_score <= 0:
        return ("SEARCH", 0.30, "fallback")

    margin = best_score - runner_up
    confidence = 0.62 + (0.08 * min(best_score, 3)) + (0.04 * min(margin, 2))
    return (best_type, max(0.0, min(0.88, confidence)), "pattern")


def _load_config(config_path: Path) -> Mapping[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_obj = _load_yaml(config_path)
    config = _as_mapping(config_obj)
    if config is None:
        raise ValueError("config.yaml must be a dictionary at top level")
    return config


def _configured_bins(config: Mapping[str, object]) -> set[str]:
    organizations = _as_mapping(config.get("organizations"))
    if organizations is None:
        return set()
    bins_obj = organizations.get("bins")
    if not isinstance(bins_obj, Sequence) or isinstance(bins_obj, (str, bytes, bytearray)):
        return set()
    return {str(item).strip() for item in bins_obj if str(item).strip()}


def _configured_model(config: Mapping[str, object]) -> str | None:
    llm = _as_mapping(config.get("llm"))
    if llm is None:
        return None
    model = str(llm.get("model", "")).strip()
    return model or None


def _configured_timeout(config: Mapping[str, object]) -> int:
    database = _as_mapping(config.get("database"))
    clickhouse = _as_mapping(database.get("clickhouse")) if database is not None else None
    value = clickhouse.get("timeout") if clickhouse is not None else None
    parsed = int(_to_float(value, default=float(DEFAULT_TIMEOUT_SECONDS)))
    return parsed if parsed > 0 else DEFAULT_TIMEOUT_SECONDS


def _get_config(config_path: Path) -> Mapping[str, object]:
    return _load_config(config_path)


def _get_llm_client(config: Mapping[str, object]) -> object:
    llm_module = importlib.import_module("src.llm.client")
    llm_client_factory = _module_attr(llm_module, "LLMClient")
    model = _configured_model(config)
    timeout = _configured_timeout(config)
    kwargs: dict[str, object] = {"timeout": timeout}
    if model:
        kwargs["model"] = model
    return _build_instance(llm_client_factory, **kwargs)


def _llm_messages(query: str) -> list[dict[str, str]]:
    system_prompt = (
        "You are a strict procurement query classifier. "
        "Classify into exactly one query_type from: SEARCH, COMPARISON, ANALYTICS, "
        "ANOMALY_DETECTION, FAIRNESS. Detect language (kazakh, russian, unknown). "
        "Extract entities: bins (12 digits), dates, enstr_codes. "
        "Dates must be normalized to these exact formats: YYYY-MM-DD for exact day, "
        "YYYY-MM for month+year, YYYY for year-only. "
        "If the query contains month+year, do not also return year-only for that same year. "
        "Respond with JSON only."
    )
    user_prompt = (
        "Classify this procurement query and return JSON object with fields "
        "query_type, language, entities{bins,dates,enstr_codes}, confidence (0..1).\n"
        "Date extraction rules:\n"
        "- exact date like 01.01.2025 or 2025-01-01 -> 2025-01-01\n"
        "- month+year like 'в феврале 2025' -> 2025-02\n"
        "- year-only like 'в 2025 году' -> 2025\n"
        "- keep all relevant dates, but never include a year-only date when a month/date from the same year exists\n"
        f"Query: {query}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_llm_response(payload: str) -> _PartialClassification:
    parsed_obj = cast(object, json.loads(_extract_json_object(payload)))
    if not isinstance(parsed_obj, Mapping):
        raise ValueError("LLM response JSON must be an object")
    parsed_map = cast(Mapping[str, object], parsed_obj)

    query_type = _to_query_type(parsed_map.get("query_type"))
    language = _to_language(parsed_map.get("language"))
    entities = _to_entities(parsed_map.get("entities"))
    confidence = _clamp_confidence(parsed_map.get("confidence"), default=0.9)

    result: _PartialClassification = {"entities": entities, "confidence": confidence}
    if query_type is not None:
        result["query_type"] = query_type
    if language is not None:
        result["language"] = language
    return result


def _classify_with_llm(
    query: str,
    config: Mapping[str, object],
) -> _PartialClassification | None:
    try:
        client = _get_llm_client(config)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("llm_client_init_failed error=%s", type(error).__name__)
        return None

    try:
        typed_client = cast(_LLMClientProtocol, client)
        response = typed_client.chat(
            messages=_llm_messages(query),
            temperature=0.0,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        if not response.strip():
            return None
        return _parse_llm_response(response)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("llm_classification_failed error=%s", type(error).__name__)
        return None


def classify(
    query: str,
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    use_llm: bool = True,
) -> ClassificationResult:
    config_path_obj = Path(config_path)
    config = _get_config(config_path_obj)
    bins = _configured_bins(config)

    direct_entities = extract_entities(query, bins)
    language_from_rules = detect_language(query)

    if use_llm:
        llm_result = _classify_with_llm(query, config)
        if llm_result is not None:
            llm_type = llm_result.get("query_type")
            if llm_type is None:
                llm_type = _pattern_query_type(query)[0]

            llm_language = llm_result.get("language")
            if llm_language is None:
                llm_language = language_from_rules

            llm_entities = llm_result.get("entities", _empty_entities())
            merged_entities = _merge_entities(direct_entities, llm_entities)
            confidence = _clamp_confidence(llm_result.get("confidence"), default=0.90)

            return {
                "query_type": llm_type,
                "language": llm_language,
                "entities": merged_entities,
                "confidence": confidence,
                "method": "llm",
            }

    query_type, confidence, method = _pattern_query_type(query)
    return {
        "query_type": query_type,
        "language": language_from_rules,
        "entities": direct_entities,
        "confidence": confidence,
        "method": method,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Procurement query classifier")
    _ = parser.add_argument("--query", type=str, required=True, help="Query text")
    _ = parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    _ = parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Enable LLM-based classification (default)",
    )
    _ = parser.add_argument(
        "--no-llm",
        action="store_false",
        dest="use_llm",
        help="Disable LLM and use only pattern fallback",
    )
    _ = parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args_map = cast(dict[str, object], vars(args))

    logging.basicConfig(
        level=logging.DEBUG if bool(args_map.get("verbose")) else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    query_text = str(args_map.get("query", ""))
    use_llm = bool(args_map.get("use_llm"))
    LOGGER.info("classifying_query length=%d use_llm=%s", len(query_text), use_llm)

    try:
        result = classify(
            query=query_text,
            config_path=cast(Path, args_map.get("config", DEFAULT_CONFIG_PATH)),
            use_llm=use_llm,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001
        LOGGER.error("classification_failed error=%s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
