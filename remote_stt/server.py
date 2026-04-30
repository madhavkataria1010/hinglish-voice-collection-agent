"""
Remote Whisper STT server.

Designed to run on a GPU box (e.g. NVIDIA A40) and serve the local agent
via a tiny HTTP contract: one POST per user turn, WAV bytes in, JSON out.

Why this exists: faster-whisper large-v3 on Mac CPU int8 was ~9-10s per
turn, blowing the <1s perceived-latency target. The same model on A40
float16 is ~50-150ms. We keep the rest of the pipeline (mic, VAD, LLM,
TTS) local so audio stays on the laptop and only the heavy STT inference
goes over the wire.

Run: uvicorn server:app --host 0.0.0.0 --port 8765
Tunnel from laptop: ssh -L 8765:localhost:8765 a40
"""
from __future__ import annotations

import io
import os
import time
import wave

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
DEFAULT_LANG = os.getenv("WHISPER_LANGUAGE", "hi") or None

app = FastAPI()
print(f"Loading {MODEL_NAME} on {DEVICE} ({COMPUTE_TYPE})...", flush=True)
_t0 = time.time()
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"Loaded in {time.time() - _t0:.1f}s", flush=True)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE, "compute": COMPUTE_TYPE}


@app.post("/transcribe")
async def transcribe(request: Request) -> JSONResponse:
    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty body"}, status_code=400)

    language = request.query_params.get("language", DEFAULT_LANG) or None
    if language == "":
        language = None

    try:
        with wave.open(io.BytesIO(body), "rb") as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except Exception as e:
        return JSONResponse({"error": f"wav decode: {e}"}, status_code=400)

    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16_000:
        ratio = 16_000 / sr
        idx = (np.arange(int(len(arr) * ratio)) / ratio).astype(np.int64)
        idx = np.clip(idx, 0, len(arr) - 1)
        arr = arr[idx]

    duration = len(arr) / 16_000.0
    t0 = time.time()
    segments, info = model.transcribe(
        arr,
        language=language,
        task="transcribe",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        without_timestamps=True,
        word_timestamps=False,
    )
    segs = list(segments)
    text = "".join(s.text for s in segs).strip()
    latency = time.time() - t0
    avg_lp = float(np.mean([s.avg_logprob for s in segs])) if segs else -1.0
    no_speech = float(np.mean([s.no_speech_prob for s in segs])) if segs else 1.0

    return JSONResponse(
        {
            "text": text,
            "language": info.language or "unknown",
            "avg_logprob": avg_lp,
            "no_speech_prob": no_speech,
            "duration_s": duration,
            "latency_s": latency,
        }
    )
