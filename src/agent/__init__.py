import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from src.agent.classifier import ClassificationResult, Entities
    from src.agent.explainability import (
        DataLineage,
        calculate_confidence,
        count_entity_types,
        extract_enstr_codes,
        format_methodology,
        generate_data_lineage,
        generate_limitations,
    )
    from src.agent.response_generator import GeneratedResponse

    classify: Callable[..., ClassificationResult]
    generate: Callable[..., GeneratedResponse]
    generate_response: Callable[..., GeneratedResponse]


def __getattr__(name: str) -> object:
    if name in {"classify", "ClassificationResult", "Entities"}:
        module = importlib.import_module("src.agent.classifier")
        return cast(object, getattr(module, name))
    if name in {"generate_response", "generate", "GeneratedResponse"}:
        module = importlib.import_module("src.agent.response_generator")
        return cast(object, getattr(module, name))
    if name in {
        "calculate_confidence",
        "generate_data_lineage",
        "format_methodology",
        "generate_limitations",
        "extract_enstr_codes",
        "count_entity_types",
        "DataLineage",
    }:
        module = importlib.import_module("src.agent.explainability")
        return cast(object, getattr(module, name))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "classify",
    "ClassificationResult",
    "Entities",
    "generate_response",
    "generate",
    "GeneratedResponse",
    "calculate_confidence",
    "generate_data_lineage",
    "format_methodology",
    "generate_limitations",
    "extract_enstr_codes",
    "count_entity_types",
    "DataLineage",
]
