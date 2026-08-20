# syntax=docker/dockerfile:1.7

FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund

COPY frontend/ ./
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build


FROM python:3.13-slim-bookworm AS python-builder

ARG POETRY_VERSION=2.4.1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    FASTEMBED_CACHE_PATH=/opt/fastembed

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --only main --no-ansi

# The adapter references this public base model. Baking it into the image
# removes a network dependency from Cloud Run cold starts.
RUN .venv/bin/python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('cross-encoder/ms-marco-MiniLM-L6-v2')"

# Hybrid retrieval needs the BM25 sparse model at query time. Cache it here so
# runtime can stay offline (HF_HUB_OFFLINE=1).
RUN mkdir -p /opt/fastembed \
    && .venv/bin/python -c \
    "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"


FROM python:3.13-slim-bookworm AS runtime

ENV PATH=/app/.venv/bin:${PATH} \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    TAXBOT_STATIC_DIR=/app/static \
    TAXBOT_RERANKER_MODEL_PATH=/app/scripts/finetuned_model \
    HF_HOME=/opt/huggingface \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 taxbot \
    && useradd --system --uid 10001 --gid taxbot --home-dir /app --shell /usr/sbin/nologin taxbot

WORKDIR /app
COPY --from=python-builder --chown=taxbot:taxbot /app/.venv ./.venv
COPY --from=python-builder --chown=taxbot:taxbot /opt/huggingface /opt/huggingface
COPY --from=python-builder --chown=taxbot:taxbot /opt/fastembed /opt/fastembed
COPY --chown=taxbot:taxbot src/ ./src/
COPY --chown=taxbot:taxbot scripts/finetuned_model/ ./scripts/finetuned_model/
COPY --from=frontend-builder --chown=taxbot:taxbot /build/frontend/dist/ ./static/

USER 10001:10001
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port \"${PORT}\""]
