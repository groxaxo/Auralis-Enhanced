FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install setuptools setuptools_scm && \
    python setup.py install

ENTRYPOINT [ "auralis.openai" ]

CMD [ "--host", "127.0.0.1", "--port", "8000", "--model", "AstraMindAI/xttsv2", "--gpt_model", "AstraMindAI/xtts2-gpt", "--max_concurrency", "8", "--vllm_logging_level", "warn" ]
