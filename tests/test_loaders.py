"""Tests for loaders: markdown/html normalization and dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_revops.loaders import load_file, load_markdown


def test_markdown_strips_syntax(tmp_path: Path):
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nSome **bold** and a [link](http://x).\n", encoding="utf-8")
    doc = load_markdown(md)
    assert "Title" in doc.text
    assert "**" not in doc.text
    assert doc.doc_type == "markdown"
    assert doc.doc_id == "doc"


def test_html_removes_scripts(tmp_path: Path):
    html = tmp_path / "page.html"
    html.write_text(
        "<html><body><p>Keep this.</p><script>drop()</script></body></html>",
        encoding="utf-8",
    )
    doc = load_file(html)
    assert "Keep this." in doc.text
    assert "drop()" not in doc.text


def test_unsupported_extension_raises(tmp_path: Path):
    bad = tmp_path / "file.xyz"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(bad)
