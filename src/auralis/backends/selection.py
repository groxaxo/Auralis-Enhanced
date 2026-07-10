"""Backend discovery and selection for Auralis."""

from __future__ import annotations

import os
import platform
from typing import Final

_BACKEND_ALIASES: Final[dict[str, str]] = {
    "auto": "auto",
    "cuda": "vllm",
    "gpu": "vllm",
    "vllm": "vllm",
    "mlx": "mlx",
    "metal": "mlx",
    "apple": "mlx",
}


def is_apple_silicon() -> bool:
    """Return ``True`` when running natively on an Apple Silicon Mac."""

    machine = platform.machine().lower()
    return platform.system() == "Darwin" and machine in {"arm64", "aarch64"}


def resolve_backend(requested: str | None = "auto") -> str:
    """Resolve a backend name, including environment and platform defaults.

    ``AURALIS_BACKEND`` takes precedence when the caller leaves the backend as
    ``None`` or ``auto``. On Apple Silicon, automatic selection chooses MLX;
    other platforms retain the existing vLLM backend.
    """

    value = (requested or "auto").strip().lower()
    if value == "auto":
        value = os.getenv("AURALIS_BACKEND", "auto").strip().lower()

    try:
        normalized = _BACKEND_ALIASES[value]
    except KeyError as exc:
        choices = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(
            f"Unsupported Auralis backend {value!r}. Choose one of: {choices}."
        ) from exc

    if normalized == "auto":
        return "mlx" if is_apple_silicon() else "vllm"
    return normalized
