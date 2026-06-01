FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/

RUN uv pip install --system -e "."

ENTRYPOINT ["Scroll"]
