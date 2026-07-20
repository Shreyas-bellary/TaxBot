"""Production frontend mounting for the FastAPI application."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def mount_frontend(app: FastAPI, static_dir: Path | None) -> bool:
    """Mount a built Vite frontend at ``/`` when a directory is configured.

    The mount must happen after all API routes are registered so ``/v1/*`` and
    health endpoints retain precedence. Returning a boolean keeps startup and
    tests explicit without reaching into Starlette's private routing state.
    """

    if static_dir is None:
        return False

    resolved = static_dir.expanduser().resolve()
    index = resolved / "index.html"
    if not index.is_file():
        raise RuntimeError(f"Frontend index not found: {index}")

    app.mount("/", StaticFiles(directory=resolved, html=True), name="frontend")
    return True
