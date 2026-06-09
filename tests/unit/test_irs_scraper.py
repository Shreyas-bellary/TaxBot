"""Deterministic unit tests for the IRS Drupal Views AJAX scraper."""

from __future__ import annotations

import httpx
import pytest
import respx

from ingestion.irs_scraper import IRSAJAXClient, _extract_insert_html

_LISTING_HTML_PAGE_0 = """
<table>
  <tbody>
    <tr>
      <td><a href="/pub/irs-pdf/f1040.pdf">Form 1040</a></td>
      <td>U.S. Individual Income Tax Return</td>
      <td>2024</td>
      <td>01/15/2025</td>
    </tr>
    <tr>
      <td><a href="/pub/irs-pdf/p17sp.pdf">Publication 17 (SP)</a></td>
      <td>Your Federal Income Tax (Spanish Version)</td>
      <td>2024</td>
      <td>01/20/2025</td>
    </tr>
    <tr>
      <td><a href="https://www.irs.gov/pub/irs-pdf/i1040gi.pdf">Instructions for Form 1040</a></td>
      <td>Instructions for Form 1040</td>
      <td>2024</td>
      <td>01/20/2025</td>
    </tr>
  </tbody>
</table>
"""

_LISTING_HTML_EMPTY = "<table><tbody></tbody></table>"


def _ajax_payload(html: str) -> list[dict[str, object]]:
    return [
        {"command": "settings", "data": {}},
        {"command": "insert", "method": "replaceWith", "data": html},
    ]


def test_extract_insert_html_handles_array_and_object_payloads() -> None:
    array_payload = _ajax_payload(_LISTING_HTML_PAGE_0)
    assert "Form 1040" in _extract_insert_html(array_payload, page=0)

    object_payload = {"commands": array_payload}
    assert "Form 1040" in _extract_insert_html(object_payload, page=0)


@pytest.mark.asyncio
async def test_iter_documents_drops_multilingual_and_paginates() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as mock:
        mock.route(host="www.irs.gov").mock(
            side_effect=[
                httpx.Response(200, json=_ajax_payload(_LISTING_HTML_PAGE_0)),
                httpx.Response(200, json=_ajax_payload(_LISTING_HTML_EMPTY)),
            ]
        )

        async with IRSAJAXClient() as client:
            client._settings = client._settings.model_copy(  # type: ignore[attr-defined]
                update={"irs_request_throttle_seconds": 0.0}
            )
            results = [
                metadata
                async for metadata in client.iter_documents(drop_multilingual=True)
            ]

    titles = [r.doc_title for r in results]
    assert "U.S. Individual Income Tax Return" in titles
    assert "Instructions for Form 1040" in titles
    assert all("version" not in t.lower() for t in titles)


@pytest.mark.asyncio
async def test_iter_documents_retries_on_transport_failure() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as mock:
        mock.route(host="www.irs.gov").mock(
            side_effect=[
                httpx.ConnectError("boom"),
                httpx.Response(200, json=_ajax_payload(_LISTING_HTML_PAGE_0)),
                httpx.Response(200, json=_ajax_payload(_LISTING_HTML_EMPTY)),
            ]
        )

        async with IRSAJAXClient() as client:
            client._settings = client._settings.model_copy(  # type: ignore[attr-defined]
                update={
                    "irs_request_throttle_seconds": 0.0,
                    "irs_max_retries": 2,
                }
            )
            results = [m async for m in client.iter_documents()]
    assert any(r.doc_number.lower().startswith("form") for r in results)


def test_parse_rows_rejects_wrong_column_count() -> None:
    bad_html = """
    <table><tr><td>only-one-cell</td></tr></table>
    """
    rows = list(IRSAJAXClient._parse_rows(bad_html))
    assert rows == []


def test_parse_rows_normalises_relative_urls() -> None:
    rows = list(IRSAJAXClient._parse_rows(_LISTING_HTML_PAGE_0))
    assert any(str(r.pdf_url).startswith("https://www.irs.gov/") for r in rows)
