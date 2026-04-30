FROM python:3.11-slim

# Audio + ffmpeg + build deps for faster-whisper / soundfile / sounddevice
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    libasound2 \
    alsa-utils \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Use uv for fast deterministic installs
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml --extra eval

COPY . .

ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=large-v3
ENV WHISPER_COMPUTE_TYPE=int8

CMD ["python", "agent.py"]
