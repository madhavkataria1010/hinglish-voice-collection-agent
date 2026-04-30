# syntax=docker/dockerfile:1
FROM python:3.12-slim

# System deps:
#   ffmpeg, libsndfile1 — audio I/O / WAV decode for the eval harness
#   libportaudio2, libasound2 — only needed if you exec into LocalAudioTransport
#       mode (host /dev/snd). The default web entry point doesn't use them.
#   build-essential, git — wheels build for ctranslate2 / numpy if pre-built
#       wheels aren't available for the platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    libasound2 \
    alsa-utils \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv handles the lockfile and is dramatically faster than pip for our deps.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy lockfile + pyproject first so the layer caches when source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Now bring in the app source.
COPY . .

ENV PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=large-v3 \
    WHISPER_COMPUTE_TYPE=int8 \
    WHISPER_BACKEND=local \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=7860

# Browser entry point — open http://<host>:7860/ once the container is up.
# This avoids mic passthrough complications (especially on macOS Docker).
# To run the local-mic agent instead:
#     docker run ... <image> uv run python agent.py
EXPOSE 7860
CMD ["uv", "run", "python", "agent_web.py"]
