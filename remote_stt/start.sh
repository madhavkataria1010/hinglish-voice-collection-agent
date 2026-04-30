#!/bin/bash
# Launches the remote Whisper STT server.
# faster-whisper needs cuDNN/cuBLAS shared libs on LD_LIBRARY_PATH; the pip
# wheels install them under the venv's site-packages, so we point at those.
set -e
cd "$(dirname "$0")"
SITE=$(./.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH=$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
export WHISPER_MODEL=${WHISPER_MODEL:-large-v3}
export WHISPER_DEVICE=${WHISPER_DEVICE:-cuda}
export WHISPER_COMPUTE_TYPE=${WHISPER_COMPUTE_TYPE:-float16}
export WHISPER_LANGUAGE=${WHISPER_LANGUAGE:-hi}
exec ./.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765
