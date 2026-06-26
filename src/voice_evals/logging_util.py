"""Structured logging setup using rich (design spec §12.1)."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging once, preferring a rich handler when available."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = level or os.environ.get("VOICE_EVALS_LOG", "INFO")
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        fmt = "%(message)s"
    except Exception:  # pragma: no cover - rich is a base dep but stay defensive
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=lvl, format=fmt, handlers=[handler], force=True)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
