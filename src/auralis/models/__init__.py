"""Built-in Auralis models.

Model modules are loaded lazily so importing :mod:`auralis` does not require
vLLM. This is essential for the optional MLX backend on macOS.
"""

from __future__ import annotations

from typing import Any

__all__ = ["XTTSv2Engine", "load_builtin_models"]


def load_builtin_models() -> None:
    """Import built-in vLLM model modules and populate the model registry."""

    from .xttsv2 import XTTSv2Engine  # noqa: F401


def __getattr__(name: str) -> Any:
    if name == "XTTSv2Engine":
        from .xttsv2 import XTTSv2Engine

        return XTTSv2Engine
    raise AttributeError(name)
