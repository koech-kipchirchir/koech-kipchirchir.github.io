# AIOS Production Dockerfile — multi-stage build
# Build:  docker build -f deploy/Dockerfile -t aios:latest .
# Run:    docker run -p 8000:8000 -v aios_data:/data aios:latest

# ---- Stage 1: Base ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIOS_ENVIRONMENT=production \
    AIOS_LOG_DIR=/var/log/aios

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Application code
COPY aios_core/ ./aios_core/
COPY api/ ./api/
COPY agents/ ./agents/
COPY models/ ./models/
COPY deploy/ ./deploy/

# Logs and data volumes
VOLUME ["/var/log/aios", "/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python deploy/scripts/healthcheck.py --url http://localhost:8000/v1/live

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info", "--proxy-headers", "--forwarded-allow-ips", "*"]
