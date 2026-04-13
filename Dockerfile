FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: gcc for any wheel fallbacks (pyarrow has wheels; keep slim).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install deps first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --upgrade pip && pip install .

# Data lives on a mounted volume in production.
RUN mkdir -p /app/data/raw /app/data/reports
ENV POLYMARKET_EDGE_DATA=/app/data

EXPOSE 8000

# Default: serve the web UI. Override CMD to run ingest/label/backtest instead.
CMD ["polymarket-edge", "web", "--host", "0.0.0.0", "--port", "8000"]
