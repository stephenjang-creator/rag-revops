"""Loaders: turn raw files (PDF / Markdown / HTML) into normalized plain text.

Each loader returns a `LoadedDocument` with the extracted text plus lightweight
source metadata that flows all the way through to citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from pypdf import PdfReader

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class LoadedDocument:
    doc_id: str          # stable id, e.g. the filename stem
    source_path: str     # where it came from
    doc_type: str        # pdf | markdown | html
    text: str            # normalized plain text
    metadata: dict = field(default_factory=dict)


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> LoadedDocument:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = _normalize("\n\n".join(pages))
    return LoadedDocument(
        doc_id=path.stem,
        source_path=str(path),
        doc_type="pdf",
        text=text,
        metadata={"n_pages": len(reader.pages)},
    )


def load_markdown(path: Path) -> LoadedDocument:
    raw = path.read_text(encoding="utf-8")
    # Render to HTML then strip tags — a cheap way to drop markdown syntax
    # while keeping readable prose intact.
    html = MarkdownIt().render(raw)
    text = _normalize(BeautifulSoup(html, "html.parser").get_text("\n"))
    return LoadedDocument(
        doc_id=path.stem,
        source_path=str(path),
        doc_type="markdown",
        text=text,
    )

def load_text(path: Path) -> LoadedDocument:
    text = _normalize(path.read_text(encoding="utf-8"))
    return LoadedDocument(
        doc_id=path.stem,
        source_path=str(path),
        doc_type="text",
        text=text,
    )

def load_html(path: Path) -> LoadedDocument:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = _normalize(soup.get_text("\n"))
    return LoadedDocument(
        doc_id=path.stem,
        source_path=str(path),
        doc_type="html",
        text=text,
    )


_DISPATCH = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".txt": load_text,
}

SUPPORTED_EXTENSIONS = tuple(_DISPATCH.keys())


def load_file(path: Path) -> LoadedDocument:
    loader = _DISPATCH.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix} ({path})")
    return loader(path)


def load_directory(directory: Path) -> list[LoadedDocument]:
    """Load every supported file under `directory` (recursively)."""
    docs: list[LoadedDocument] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in _DISPATCH:
            docs.append(load_file(path))
    return docs
