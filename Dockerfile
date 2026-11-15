# EchoCount Production Dockerfile (CPU default; optional GPU instructions below)
#
# Build & Run (CPU):
#   docker build -t echocount:latest .
#   docker run --rm -p 8080:8080 -e ECHOCOUNT_DEV=0 echocount:latest
#
# Build & Run (NVIDIA GPU with CUDA 12):
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime -t echocount:gpu .
#   docker run --rm --gpus all -p 8080:8080 -e ECHOCOUNT_DEV=0 echocount:gpu

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    ECHOCOUNT_DEV=0 \
    MAX_UPLOAD_BYTES=41943040 \
    MAX_DURATION_SECONDS=600 \
    JOB_TTL_SECONDS=1800 \
    RATE_LIMIT_PER_MINUTE=120 \
    SEPARATION_WORKERS=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads/jobs checkpoints models evaluation

EXPOSE 8080
CMD ["sh", "-c", "exec gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT app:app"]
