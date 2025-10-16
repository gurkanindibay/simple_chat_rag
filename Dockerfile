# syntax=docker/dockerfile:1.4
FROM python:3.11-slim
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first so this layer is cached unless requirements change
COPY requirements-docker.txt requirements-docker.txt
# Use BuildKit cache for pip to avoid re-downloading packages on every rebuild
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install -r requirements-docker.txt

COPY . .

EXPOSE 8000

CMD ["./docker/wait-for-db.sh", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
