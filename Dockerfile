FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential portaudio19-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install ".[cuda,server]"

ENTRYPOINT ["auralis.openai"]
CMD ["--backend", "vllm", "--host", "0.0.0.0", "--port", "8000", "--model", "AstraMindAI/xttsv2", "--gpt_model", "AstraMindAI/xtts2-gpt", "--max_concurrency", "8", "--vllm_logging_level", "warn"]
