"""Fetch the public demo corpus into data/raw/.

Two sources:
  1. CUAD (CC BY 4.0) — real commercial contracts + clause annotations. We pull a
     small sample of the plain-text contracts for the demo corpus. The full set
     and its annotations are used later to seed the Phase 3 evaluation set.
  2. Publicly published SaaS agreements / SLAs — copyrighted, so we DO NOT vendor
     the files into the repo. This script downloads them locally on demand and
     records attribution. Review each source's terms before redistributing.

Usage:
    python scripts/download_corpus.py               # fetch everything
    python scripts/download_corpus.py --skip-cuad   # only the linked agreements
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

RAW_DIR = Path("data/raw")

# CUAD full corpus (zip). Hosted by The Atticus Project. CC BY 4.0.
CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"

# Publicly posted SaaS/legal PDFs. These are linked, not committed. Verify terms.
PUBLIC_AGREEMENTS: list[tuple[str, str]] = [
    (
        "cooley_saas_model_agreement.pdf",
        "https://www.acc.com/sites/default/files/program-materials/upload/"
        "Cooley%20SaaS%20Agreement%20ACC%20Form.pdf",
    ),
]


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        print(f"  ✓ {dest}")
        return True
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"  ✗ {url}\n      {exc}", file=sys.stderr)
        return False


def fetch_cuad() -> None:
    print("CUAD (CC BY 4.0):")
    dest = RAW_DIR / "cuad" / "data.zip"
    if _download(CUAD_URL, dest):
        print("    Note: unzip and select a sample of full_contract_txt/*.txt")
        print("    for the demo corpus; keep master_clauses.csv for Phase 3 eval.")


def fetch_agreements() -> None:
    print("Public SaaS agreements (linked, review terms before redistributing):")
    for name, url in PUBLIC_AGREEMENTS:
        _download(url, RAW_DIR / "agreements" / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cuad", action="store_true")
    parser.add_argument("--skip-agreements", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_cuad:
        fetch_cuad()
    if not args.skip_agreements:
        fetch_agreements()
    print("\nDone. See docs/DATA_PROVENANCE.md for attribution and licensing.")


if __name__ == "__main__":
    main()
