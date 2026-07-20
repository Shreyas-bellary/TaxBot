"""Tests for production frontend mounting."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.static import mount_frontend


def test_static_frontend_disabled() -> None:
    app = FastAPI()

    assert mount_frontend(app, None) is False


def test_static_frontend_serves_index_after_api_routes(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>TaxBot</h1>", encoding="utf-8")
    app = FastAPI()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    assert mount_frontend(app, tmp_path) is True

    with TestClient(app) as client:
        assert client.get("/").text == "<h1>TaxBot</h1>"
        assert client.get("/healthz").json() == {"status": "ok"}


def test_static_frontend_requires_index(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Frontend index not found"):
        mount_frontend(FastAPI(), tmp_path)
