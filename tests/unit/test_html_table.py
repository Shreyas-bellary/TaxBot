"""Tests for the HTML-to-Markdown table converter."""

from __future__ import annotations

from ingestion.unstructured_parser import html_table_to_markdown


def test_html_table_to_markdown_emits_header_separator() -> None:
    html = """
    <table>
      <thead>
        <tr><th>Filing Status</th><th>Bracket</th><th>Rate</th></tr>
      </thead>
      <tbody>
        <tr><td>Single</td><td>$0 - $11,600</td><td>10%</td></tr>
        <tr><td>Single</td><td>$11,601 - $47,150</td><td>12%</td></tr>
      </tbody>
    </table>
    """
    md = html_table_to_markdown(html)
    lines = md.splitlines()
    assert lines[0].startswith("| Filing Status |")
    assert lines[1].count("---") == 3
    assert any("$11,601" in line for line in lines)


def test_html_table_to_markdown_pipes_are_escaped() -> None:
    html = "<table><tr><th>x|y</th></tr><tr><td>a|b</td></tr></table>"
    md = html_table_to_markdown(html)
    assert "x\\|y" in md
    assert "a\\|b" in md


def test_html_table_to_markdown_empty_input() -> None:
    assert html_table_to_markdown("") == ""
