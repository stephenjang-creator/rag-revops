"""Deal Desk Helper — grounded, citation-enforced contract analysis over public data."""

# Load .env into os.environ as early as possible — before any submodule
# (notably observability.py) reads the environment at import time. This runs
# when the first `rag_revops.*` import fires, so LANGFUSE_* and other vars from
# .env are present regardless of how the process was launched (CLI, Streamlit,
# pytest). python-dotenv ships transitively via pydantic-settings, so no new
# dependency. Absent .env → no-op; real environment variables always win
# (override=False), so CI/Streamlit secrets aren't clobbered.
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(override=False)
except Exception:  # pragma: no cover - dotenv missing or unreadable .env
    pass

__version__ = "0.1.0"
