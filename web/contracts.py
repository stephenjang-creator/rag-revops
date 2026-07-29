"""Parse a readable title + year out of the packed CUAD doc ids, so the
single-contract picker shows names instead of raw ids like
``041__NICELTD_2003-EX-4.5-OUTSOURCING_AGREEMENT``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_YEAR = re.compile(r"(19|20)\d{2}")


@dataclass
class ContractMeta:
    doc_id: str
    title: str
    year: int | None


def _humanize(token: str) -> str:
    # "OUTSOURCING_AGREEMENT" / "NICELTD" -> "Outsourcing Agreement" / "Niceltd"
    words = re.split(r"[_\-]+", token)
    return " ".join(w.capitalize() for w in words if w)


def parse_contract(doc_id: str) -> ContractMeta:
    # Strip the leading "NNN__" index prefix.
    body = re.sub(r"^\d+__", "", doc_id)
    year_m = _YEAR.search(body)
    year = int(year_m.group(0)) if year_m else None

    party = body[: year_m.start()] if year_m else body
    party = party.rstrip("_-")
    # Drop trailing date fragments (e.g. "..._09_09" before a 1999) so the title
    # is the party name, not the filing date.
    party = re.sub(r"(?:[_\-]\d{1,2}){1,2}$", "", party)
    # The tail after the year is usually EX-x.y-<TYPE>; keep the TYPE.
    doc_type = ""
    if year_m:
        tail = body[year_m.end():].lstrip("_-")
        # Drop an "EX-10.1" style exhibit marker if present, keep the rest.
        tail = re.sub(r"^EX[-_]?[\d.]+[-_]?", "", tail, flags=re.IGNORECASE)
        doc_type = _humanize(tail)

    name = _humanize(party)
    title = f"{name} — {doc_type}".strip(" —") if doc_type else name
    return ContractMeta(doc_id=doc_id, title=title or doc_id, year=year)


def list_contracts(doc_ids: list[str]) -> list[dict]:
    metas = [parse_contract(d) for d in doc_ids]
    return [{"doc_id": m.doc_id, "title": m.title, "year": m.year} for m in metas]
