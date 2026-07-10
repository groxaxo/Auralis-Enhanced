"""Optional inference backends exposed by Auralis."""

from .selection import is_apple_silicon, resolve_backend

__all__ = ["is_apple_silicon", "resolve_backend"]
