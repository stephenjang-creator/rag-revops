"""Assemble a small, curated demo subset for the hosted Streamlit demo.

The full corpus is too large (and partly not redistributable) to commit. For the
live demo we want a *small, self-contained, public* slice whose prebuilt Chroma
index can be committed to the repo so Streamlit Cloud (ephemeral filesystem) has
data on cold start.

This script copies a capped number of CUAD plain-text contracts (CC BY 4.0) plus
any downloaded public agreements into data/raw/demo_subset/. You then run:

    python -m rag_revops.ingest --source data/raw/demo_subset
    git add -f data/processed/chroma      # deliberate .gitignore exception

Usage:
    python scripts/build_demo_subset.py --max-contracts 18
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

RAW = Path("data/raw")
SUBSET = RAW / "demo_subset"
CUAD_TXT = RAW / "cuad" / "full_contract_txt"   # after unzipping CUAD data.zip
AGREEMENTS = RAW / "agreements"


def build(max_contracts: int) -> None:
    SUBSET.mkdir(parents=True, exist_ok=True)
    # Clear any prior subset so re-runs are deterministic.
    for f in SUBSET.iterdir():
        if f.is_file():
            f.unlink()

    copied = 0

    if CUAD_TXT.exists():
        contracts = sorted(CUAD_TXT.glob("*.txt"))[:max_contracts]
        for src in contracts:
            shutil.copy2(src, SUBSET / src.name)
            copied += 1
        print(f"Copied {len(contracts)} CUAD contracts.")
    else:
        print(f"[warn] {CUAD_TXT} not found — unzip CUAD data.zip first.")

    if AGREEMENTS.exists():
        for src in sorted(AGREEMENTS.glob("*")):
            if src.is_file():
                shutil.copy2(src, SUBSET / src.name)
                copied += 1
        print(f"Copied public agreements from {AGREEMENTS}.")

    print(f"\nDemo subset ready at {SUBSET} ({copied} files).")
    print("Next:")
    print("  python -m rag_revops.ingest --source data/raw/demo_subset")
    print("  git add -f data/processed/chroma")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-contracts", type=int, default=18)
    build(ap.parse_args().max_contracts)


if __name__ == "__main__":
    main()
