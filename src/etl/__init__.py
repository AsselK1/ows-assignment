import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import cast

from collections.abc import Sequence

from .load_ref_data import load_enstr, load_kato, load_mkei, main


def _invoke_loader(
    module_name: str,
    function_name: str,
    *args: object,
    **kwargs: object,
) -> dict[str, int]:
    module = importlib.import_module(module_name)
    loader = cast(Callable[..., dict[str, int]], getattr(module, function_name))
    signature = inspect.signature(loader)
    if "id_filter" not in signature.parameters and "id_filter" in kwargs:
        kwargs = dict(kwargs)
        _ = kwargs.pop("id_filter", None)
    return loader(*args, **kwargs)


def load_subjects(
    ows_client: object,
    clickhouse: object,
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int = 500,
    force: bool = False,
    id_filter: Sequence[int] | None = None,
) -> dict[str, int]:
    return _invoke_loader(
        "src.etl.etl_subjects",
        "load_subjects",
        ows_client,
        clickhouse,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        force=force,
        id_filter=id_filter,
    )


def load_plans(
    ows_client: object,
    clickhouse: object,
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int = 500,
    force: bool = False,
    id_filter: Sequence[int] | None = None,
) -> dict[str, int]:
    return _invoke_loader(
        "src.etl.etl_plans",
        "load_plans",
        ows_client,
        clickhouse,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        force=force,
        id_filter=id_filter,
    )


def load_announcements(
    ows_client: object,
    clickhouse: object,
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int = 500,
    force: bool = False,
    id_filter: Sequence[int] | None = None,
) -> dict[str, int]:
    return _invoke_loader(
        "src.etl.etl_announcements",
        "load_announcements",
        ows_client,
        clickhouse,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        force=force,
        id_filter=id_filter,
    )


def load_lots(
    ows_client: object,
    clickhouse: object,
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int = 500,
    force: bool = False,
    id_filter: Sequence[int] | None = None,
) -> dict[str, int]:
    return _invoke_loader(
        "src.etl.etl_lots",
        "load_lots",
        ows_client,
        clickhouse,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        force=force,
        id_filter=id_filter,
    )


def load_contracts(
    ows_client: object,
    clickhouse: object,
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int = 500,
    force: bool = False,
    id_filter: Sequence[int] | None = None,
) -> dict[str, int]:
    return _invoke_loader(
        "src.etl.etl_contracts",
        "load_contracts",
        ows_client,
        clickhouse,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        force=force,
        id_filter=id_filter,
    )


def load_contract_acts(
    ows_client: object,
    clickhouse: object,
    *,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int = 500,
    force: bool = False,
    id_filter: Sequence[int] | None = None,
) -> dict[str, int]:
    return _invoke_loader(
        "src.etl.etl_contract_acts",
        "load_contract_acts",
        ows_client,
        clickhouse,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        force=force,
        id_filter=id_filter,
    )


__all__ = [
    "load_enstr",
    "load_kato",
    "load_mkei",
    "load_subjects",
    "load_plans",
    "load_announcements",
    "load_lots",
    "load_contracts",
    "load_contract_acts",
    "main",
]
