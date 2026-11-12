# Reproducible environment for the pipeline and the app, built from the locked dependencies (uv.lock).
#   docker build -t probity .
#   docker run --gpus all -e HF_TOKEN -p 7860:7860 probity                 # the app on http://localhost:7860
#   docker run --gpus all -e HF_TOKEN -v "$PWD/outputs:/app/outputs" probity run   # the full pipeline
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project --extra all

COPY src ./src
COPY configs ./configs
COPY results ./results
COPY figures ./figures
COPY assets ./assets
RUN uv sync --frozen --no-dev --extra all

EXPOSE 7860
ENTRYPOINT ["probity"]
CMD ["app", "--host", "0.0.0.0", "--port", "7860"]
