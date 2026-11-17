FROM ghcr.io/astral-sh/uv:0.9.8 AS uv
FROM python:3.11-slim
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY configs ./configs
COPY scripts ./scripts
COPY app ./app
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/app/.cache/huggingface" \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["python", "scripts/run_demo.py", "--host", "0.0.0.0", "--port", "8000"]

