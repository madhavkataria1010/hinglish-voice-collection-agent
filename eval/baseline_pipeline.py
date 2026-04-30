"""
Baseline pipeline: pure-API stack (Deepgram STT + OpenAI TTS) for comparison.

NOT used in the live agent — only run during evaluation. We deliberately omit
the number-normalizer, language router, filler injector and placeholder
substitution from this path so the comparison reflects "naive commercial
wrapper" vs. our enhanced pipeline.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import httpx
import numpy as np
from loguru import logger


@dataclass
class STTResult:
    text: str
    language: str
    latency_s: float


class DeepgramSTT:
    """Minimal Deepgram pre-recorded transcribe client (sync per clip)."""

    URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set")
        self._http = httpx.AsyncClient(timeout=15.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def transcribe_wav(self, wav_path: str) -> STTResult:
        # Deepgram nova-3 is their best multilingual model with Hindi/English
        params = {"model": "nova-3", "detect_language": "true", "smart_format": "true"}
        with open(wav_path, "rb") as f:
            audio = f.read()
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }
        t0 = time.time()
        resp = await self._http.post(
            self.URL, params=params, content=audio, headers=headers
        )
        latency = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        try:
            channel = data["results"]["channels"][0]
            alt = channel["alternatives"][0]
            text = alt.get("transcript", "")
            lang = channel.get("detected_language", "unknown")
        except (KeyError, IndexError):
            text, lang = "", "unknown"
        return STTResult(text=text, language=lang, latency_s=latency)


class OpenAITTSBaseline:
    """OpenAI TTS-1 (closed). Used only for baseline pipeline comparisons."""

    URL = "https://api.openai.com/v1/audio/speech"

    def __init__(self, api_key: str | None = None, voice: str = "alloy") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.voice = voice
        self._http = httpx.AsyncClient(timeout=15.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def synthesize(self, text: str) -> tuple[bytes, float]:
        """Return (audio_bytes_pcm16_24k, ttfb_s)."""
        payload = {
            "model": "tts-1",
            "voice": self.voice,
            "input": text,
            "response_format": "pcm",  # raw 24kHz pcm16
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        t0 = time.time()
        resp = await self._http.post(self.URL, json=payload, headers=headers)
        ttfb = time.time() - t0
        resp.raise_for_status()
        return resp.content, ttfb
