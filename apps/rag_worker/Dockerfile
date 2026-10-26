FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY apps/rag_worker/requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY apps/rag_worker ./worker
RUN pip install --no-cache-dir --no-deps ./worker && useradd --uid 10001 --create-home worker
USER worker
WORKDIR /home/worker
CMD ["python", "-m", "apps.rag_worker.main", "--embedding", "/models/bge-m3", "--reranker", "/models/bge-reranker-large", "--port", "8091"]
