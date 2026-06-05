FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
COPY pyproject.toml requirements.lock README.md ./
RUN pip install --no-cache-dir -r requirements.lock
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir --no-deps -e .
RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).read()"
CMD ["uvicorn", "f1di.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
