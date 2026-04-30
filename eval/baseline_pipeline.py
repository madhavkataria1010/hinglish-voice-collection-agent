"""
Baseline pipeline: pure-API stack (commercial STT + OpenAI TTS) for comparison.

Two STT options, picked at runtime:
  • Deepgram nova-3 (preferred) when DEEPGRAM_API_KEY is set and accepted by
    the API. This is the assignment's named-example baseline.
  • OpenAI Whisper API (gpt-4o-transcribe) fallback when Deepgram isn't
    available. Same shape — closed-source, commercial, no normalizer / no
    router / no filler — so the comparison still measures "naive commercial
    wrapper" vs. our pipeline.

NOT used in the live agent — only during evaluation. We deliberately omit
the number-normalizer, language router, filler injector, and placeholder
substitution from this path.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import httpx
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


class OpenAIWhisperSTT:
    """OpenAI hosted Whisper API — fallback baseline STT.

    Used when Deepgram returns 401/403 or the key is missing. We still
    treat this as a "commercial closed-source baseline" because it's the
    OpenAI-hosted variant, not faster-whisper running locally.
    """

    URL = "https://api.openai.com/v1/audio/transcriptions"

    def __init__(self, api_key: str | None = None, model: str = "whisper-1") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.model = model
        self._http = httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def transcribe_wav(self, wav_path: str) -> STTResult:
        with open(wav_path, "rb") as f:
            audio = f.read()
        files = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": (None, self.model),
            "response_format": (None, "verbose_json"),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.time()
        resp = await self._http.post(self.URL, files=files, headers=headers)
        latency = time.time() - t0
        if resp.status_code != 200:
            return STTResult(text="", language="unknown", latency_s=latency)
        data = resp.json()
        return STTResult(
            text=data.get("text", "").strip(),
            language=data.get("language", "unknown"),
            latency_s=latency,
        )


async def make_baseline_stt() -> "DeepgramSTT | OpenAIWhisperSTT":
    """Pick whichever commercial STT is reachable.

    Tries Deepgram first (the assignment's named example). If the
    DEEPGRAM_API_KEY is missing OR returns 401/403, falls back to OpenAI's
    hosted Whisper API. We probe with a short HEAD-equivalent call (zero-
    byte POST) so we don't waste a real clip on the unauth path.
    """
    if os.getenv("DEEPGRAM_API_KEY"):
        candidate = DeepgramSTT()
        # Probe auth with a tiny empty POST. Deepgram rejects empty audio
        # with 400 ("no audio") on a valid key and 401 on an invalid one.
        try:
            async with httpx.AsyncClient(timeout=5.0) as probe:
                r = await probe.post(
                    candidate.URL,
                    headers={
                        "Authorization": f"Token {candidate.api_key}",
                        "Content-Type": "audio/wav",
                    },
                    content=b"",
                )
            if r.status_code in (401, 403):
                await candidate.aclose()
                logger.warning(
                    "DEEPGRAM_API_KEY rejected by Deepgram "
                    f"(HTTP {r.status_code}); falling back to OpenAI Whisper API."
                )
            else:
                return candidate
        except Exception as e:
            await candidate.aclose()
            logger.warning(f"Deepgram probe errored ({e}); using OpenAI Whisper API.")
    return OpenAIWhisperSTT()


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
