"""Optional Prefect integration helpers."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any

_PREFECT_INSTALL_HINT = (
    "Prefect is required for workflow runtime support. "
    "Install it with `pyna-chaos[workflow]` or `pyna-chaos[prefect]`."
)


def prefect_runtime_available() -> bool:
    """Return whether Prefect can be imported in the current environment."""
    return importlib.util.find_spec("prefect") is not None


def require_prefect() -> ModuleType:
    """Return the Prefect module or raise a clear installation error."""
    if not prefect_runtime_available():
        raise RuntimeError(_PREFECT_INSTALL_HINT)
    return importlib.import_module("prefect")


def optional_prefect() -> tuple[Any, Any]:
    """Return Prefect's ``flow`` and ``task`` callables if available."""
    prefect = require_prefect()
    return prefect.flow, prefect.task


__all__ = [
    "optional_prefect",
    "prefect_runtime_available",
    "require_prefect",
]
