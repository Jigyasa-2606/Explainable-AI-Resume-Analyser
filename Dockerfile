FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY backend ./backend

ENV PYTHONUNBUFFERED=1
ENV STAGE2_RERANKER=off
ENV USE_LTR_RERANK=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)"

CMD ["uvicorn", "backend.backend_api:app", "--host", "0.0.0.0", "--port", "8000"]
