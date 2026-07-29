# Deal Desk Helper — FastAPI web service for Render.
# The demo Chroma index is committed under data/processed/chroma, so it's baked
# into the image; cold starts don't re-ingest. Keys are bring-your-own (per
# request), so no API keys are needed in the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# chroma-hnswlib / some transitive deps may compile if a wheel isn't available.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install deps first (better layer caching) — needs pyproject + src for the
# editable install to resolve the package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[web]"

# App code + the committed demo index + versioned config.
COPY . .

EXPOSE 8000
# `python -m uvicorn` puts the repo root on sys.path so the `web` package and the
# relative data/ + config/ paths resolve from WORKDIR.
CMD ["sh", "-c", "python -m uvicorn web.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
