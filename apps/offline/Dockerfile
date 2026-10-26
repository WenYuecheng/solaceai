FROM node:22-alpine AS frontend-builder
WORKDIR /workspace
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml ./frontend/
RUN cd frontend && pnpm install --frozen-lockfile
COPY frontend ./frontend
RUN cd frontend && pnpm build

FROM python:3.12-slim-bookworm AS builder
WORKDIR /build
RUN python -m venv /opt/venv
COPY apps/offline/requirements.lock ./requirements.lock
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN /opt/venv/bin/pip install --timeout 120 --retries 10 --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements.lock
# Docling uses a separate dependency environment (the API pins settings 2.7.1).
COPY deploy/requirements/requirements-documents.txt ./requirements-documents.txt
RUN python -m venv /opt/documents && /opt/documents/bin/pip install --timeout 120 --retries 10 --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements-documents.txt
COPY packages/core ./packages/core
COPY apps/offline ./apps/offline
COPY --from=frontend-builder /workspace/packages/core/app/static ./packages/core/app/static
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps ./packages/core ./apps/offline

FROM python:3.12-slim-bookworm
ENV DOCUMENT_CONVERTER_PYTHON=/opt/documents/bin/python PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH" SOLACEAI_ROOT=/app
RUN groupadd --gid 10001 solaceai && useradd --uid 10001 --gid 10001 --create-home solaceai
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/documents /opt/documents
COPY THIRD_PARTY_NOTICES.md ./THIRD_PARTY_NOTICES.md
RUN mkdir -p /app/data/offline && chown -R solaceai:solaceai /app
USER solaceai
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/actuator/health', timeout=3)"
CMD ["uvicorn", "apps.offline.main:app", "--host", "0.0.0.0", "--port", "8080"]
