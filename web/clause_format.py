"""Turn the drafter's Markdown into (a) clean formatted HTML for display and
(b) order-form-ready plain text for copy/paste.

The clause drafter (`rag_revops.clause_drafting`) returns whatever Markdown the
model writes — headings (``##``), bold (``**``), blockquotes (``>``), horizontal
rules, and sometimes literal ``&nbsp;`` entities. Rendering that raw looked like
ANSI soup. Here we:

- ``to_html`` — render a SAFE Markdown subset (no raw HTML, no links/images) and
  wrap inline ``[n]`` citation markers in ``<sup>``.
- ``to_plain`` — strip all of it to the clause language a Deal Desk would paste
  into an order form or Word: no Markdown syntax, no ``&nbsp;``, no ``[n]``
  citations, and no deal-desk-note blockquote (that's commentary, not clause).
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

from rag_revops.clause_drafting import strip_citations

# A deliberately small, safe subset: structure + emphasis, but no links, images,
# raw HTML, or autolinks (the source is model output, so we don't trust it to
# render arbitrary anchors). `breaks` keeps the model's line breaks visible.
_MD = MarkdownIt("zero", {"breaks": True}).enable(
    ["heading", "paragraph", "blockquote", "list", "emphasis", "hr", "newline"]
)

_NBSP = re.compile(r"&nbsp;|\xa0")
_CITE = re.compile(r"(\[\d+(?:\s*,\s*\d+)*\])")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BLOCKQUOTE_LINE = re.compile(r"^\s{0,3}>.*$", re.M)
_HR_LINE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$", re.M)
_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.M)


def to_html(draft: str) -> str:
    """Formatted, sanitized HTML for the clause panel; citations become <sup>."""
    html = _MD.render(_NBSP.sub(" ", draft))
    return _CITE.sub(r"<sup>\1</sup>", html).strip()


def to_plain(draft: str) -> str:
    """Clean clause language for pasting into an order form / Word: no Markdown,
    no &nbsp;, no [n] citations, and no deal-desk-note blockquote."""
    text = _NBSP.sub(" ", draft)
    text = _HR_LINE.sub("", text)          # horizontal rules
    text = _BLOCKQUOTE_LINE.sub("", text)  # the "> Deal Desk Note:" commentary
    text = _HEADING.sub("", text)          # keep heading text, drop the # markers
    text = _BULLET.sub("• ", text)         # normalize list bullets
    text = text.replace("**", "").replace("__", "")  # bold markers
    text = strip_citations(text)           # [n] markers + now-empty parens
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim trailing spaces per line, then overall.
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()
