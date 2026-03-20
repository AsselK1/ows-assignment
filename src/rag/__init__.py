import importlib
from typing import cast


def __getattr__(name: str) -> object:
    if name in {"VectorStore", "SearchResult"}:
        module = importlib.import_module("src.rag.vector_store")
        return cast(object, getattr(module, name))
    if name in {"HybridRetriever", "retrieve"}:
        module = importlib.import_module("src.rag.retriever")
        return cast(object, getattr(module, name))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
