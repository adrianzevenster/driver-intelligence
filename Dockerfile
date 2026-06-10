# ── stage 1: build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# ── stage 2: Python API ────────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
COPY pyproject.toml requirements.lock README.md ./
RUN pip install --no-cache-dir -r requirements.lock
RUN python -c "\
try:\
    from sentence_transformers import SentenceTransformer;\
    SentenceTransformer('all-MiniLM-L6-v2');\
    print('Embedding model cached.')\
except ImportError:\
    pass\
"
COPY src ./src
COPY data ./data
COPY migrations ./migrations
COPY alembic.ini ./
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
RUN pip install --no-cache-dir --no-deps -e .
RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /app/data/storage && chown appuser /app/data/storage
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).read()"
CMD ["sh", "-c", "alembic upgrade head && uvicorn f1di.api.main:app --host 0.0.0.0 --port 8080"]
