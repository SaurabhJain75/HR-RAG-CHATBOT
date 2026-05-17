# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Builder
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir

# Install ALL packages in one command from CPU-only index
# This avoids version conflicts between torch and torchaudio
RUN pip install \
      --no-cache-dir \
      --prefer-binary \
      --prefix=/install \
      -r requirements.txt \
      -f https://download.pytorch.org/whl/cpu/torch_stable.html


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Final Image
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PATH=/usr/local/bin:$PATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY config.py .
COPY models.py .
COPY prompts.py .
COPY rag.py .
COPY ingest.py .
COPY tools.py .
COPY agents.py .
COPY schemas.py .
COPY api.py .
COPY app.py .

# Copy HR policy documents into image
COPY data/ data/

RUN mkdir -p vectorstore .cache/sentence_transformers

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]