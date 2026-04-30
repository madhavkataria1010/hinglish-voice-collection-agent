"""
Sarvam Bulbul-v2 TTS client.

Why Sarvam: most natural Hindi/Hinglish voice in the closed-source TTS market
that also does English without a voice flip — same speaker handles both
registers, which is exactly what we need to eliminate the 2-3 second
language-switch pause described in the assignment.

Wraps Sarvam's text-to-speech REST endpoint and exposes a Pipecat TTSService.
For streaming we chunk the LLM output by sentence and pipeline the requests.
"""
from __future__ import annotations

import asyncio
import base64
import os
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from loguru import logger

# Devanagari + ASCII letters; Sarvam's "allowed languages" check rejects any
# input that has zero characters from a recognized script.
_HAS_ALPHANUM = re.compile(r"[A-Za-zऀ-ॿ0-9]")

try:
    from pipecat.frames.frames import (
        Frame,
        TTSAudioRawFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
    )
    from pipecat.services.tts_service import TTSService
    PIPECAT_AVAILABLE = True
except Exception:  # pragma: no cover
    PIPECAT_AVAILABLE = False
    TTSService = object  # type: ignore[misc,assignment]


SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
DEFAULT_VOICE = "anushka"   # Sarvam multilingual female voice
DEFAULT_MODEL = "bulbul:v2"
DEFAULT_SAMPLE_RATE = 22_050


class SarvamClient:
    """Stateless HTTP client. One client per agent process."""

    def __init__(
        self,
        api_key: str | None = None,
        voice: str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        timeout_s: float = 8.0,
    ) -> None:
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY not set")
        self.voice = voice
        self.model = model
        self.sample_rate = sample_rate
        self._http = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def synthesize(
        self,
        text: str,
        target_language_code: str = "hi-IN",
    ) -> tuple[bytes, float]:
        """
        Synthesize ``text`` and return (pcm_bytes, ttfb_seconds).

        Returns 16-bit PCM mono at ``self.sample_rate``. Sarvam responds with
        base64 WAV; we strip the 44-byte header to get raw PCM.
        """
        payload = {
            "inputs": [text],
            "target_language_code": target_language_code,
            "speaker": self.voice,
            "model": self.model,
            "speech_sample_rate": self.sample_rate,
            "enable_preprocessing": True,
        }
        headers = {"api-subscription-key": self.api_key}
        t0 = time.time()
        resp = await self._http.post(SARVAM_TTS_URL, json=payload, headers=headers)
        ttfb = time.time() - t0
        if resp.status_code != 200:
            raise RuntimeError(
                f"Sarvam TTS error {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        wav_b64 = data["audios"][0]
        wav_bytes = base64.b64decode(wav_b64)
        # WAV header is 44 bytes for PCM; skip it
        pcm = wav_bytes[44:] if wav_bytes[:4] == b"RIFF" else wav_bytes
        return pcm, ttfb


# --------------------------------------------------------------------------- #
# Pipecat integration
# --------------------------------------------------------------------------- #

if PIPECAT_AVAILABLE:

    class SarvamTTSService(TTSService):  # type: ignore[misc]
        """
        Pipecat TTS service that streams Sarvam audio to the output transport.

        We synthesize per sentence so the speaker hears the first sentence
        while later ones are still in flight. Pipecat's frame plumbing handles
        playback ordering.
        """

        def __init__(
            self,
            api_key: str | None = None,
            voice: str = DEFAULT_VOICE,
            model: str = DEFAULT_MODEL,
            sample_rate: int = DEFAULT_SAMPLE_RATE,
            target_language_code: str = "hi-IN",
            text_rewriter: "Callable[[str], str] | None" = None,
            **kwargs: Any,
        ) -> None:
            from pipecat.services.settings import TTSSettings

            kwargs.setdefault(
                "settings",
                TTSSettings(model=model, voice=voice, language=target_language_code),
            )
            super().__init__(sample_rate=sample_rate, **kwargs)
            self._client = SarvamClient(
                api_key=api_key,
                voice=voice,
                model=model,
                sample_rate=sample_rate,
            )
            self._target_lang = target_language_code
            # Authoritative placeholder substitution. Pipecat aggregates
            # streaming LLM tokens into sentence-level chunks before calling
            # run_tts, so a placeholder like "{settlement_amount}" is always
            # complete in the text we get here. Doing it at this exact point
            # — the last hop before audio synthesis — guarantees no leak.
            self._text_rewriter = text_rewriter

        def set_target_language(self, code: str) -> None:
            """Switch the *default* target_language_code at runtime.

            The runtime pick in :meth:`_pick_lang` overrides this when the
            text is unambiguously English or Devanagari; this default only
            matters for ambiguous text (e.g. transliterated Hinglish in
            Latin script).
            """
            self._target_lang = code

        def _pick_lang(self, text: str) -> str:
            """Choose hi-IN vs en-IN based on the text content.

            - Any Devanagari character → hi-IN (Sarvam handles mixed Hindi
              + English in hi-IN cleanly when Devanagari is present).
            - Otherwise → en-IN, so English names and Indian-English
              phrasing pronounce correctly.
            """
            for ch in text:
                if "ऀ" <= ch <= "ॿ":
                    return "hi-IN"
            return "en-IN"

        async def run_tts(
            self, text: str, context_id: str | None = None
        ) -> AsyncGenerator[Frame, None]:
            text = text.strip()
            if not text:
                return
            # Authoritative placeholder substitution. We MUST do it here,
            # not (only) in OutboundTurnProcessor, because:
            #   1. Pipecat's TTS service runs its own text aggregator on
            #      streaming LLM tokens, and we don't want to depend on
            #      ordering between our processor and that aggregator.
            #   2. By the time text reaches run_tts, it is sentence-level
            #      and any `{...}` placeholder is guaranteed complete.
            # Without this backstop we observed `{borrower_offer}` and
            # `{settlement_amount}` leaking into spoken audio.
            if self._text_rewriter is not None:
                rewritten = self._text_rewriter(text)
                if rewritten != text:
                    logger.debug(
                        f"TTS rewrite: {text[:60]!r} -> {rewritten[:60]!r}"
                    )
                    text = rewritten
            if not text.strip():
                return
            # Sarvam rejects any input that has zero characters from a
            # recognized script ("Input texts must contain at least one
            # character from the allowed languages."). The streaming sentence
            # splitter occasionally hands us pure-punctuation chunks like ", "
            # or "..." — drop those silently.
            if not _HAS_ALPHANUM.search(text):
                logger.debug(f"TTS skip: no speakable chars in {text!r}")
                return
            # Auto-pick target language: Devanagari → hi-IN, otherwise en-IN.
            # Forcing hi-IN on English text mangles English names ("Rajesh"
            # → "rages"); en-IN on the same Bulbul voice handles English
            # cleanly without a voice-flip.
            tts_lang = self._pick_lang(text)
            yield TTSStartedFrame()
            try:
                pcm, ttfb = await self._client.synthesize(
                    text, target_language_code=tts_lang
                )
                logger.debug(
                    f"TTS [{tts_lang}] ttfb={ttfb*1000:.0f}ms "
                    f"chars={len(text)} bytes={len(pcm)}"
                )
                # Stream in 20ms chunks so playback can start immediately
                bytes_per_20ms = int(self.sample_rate * 0.02) * 2  # 16-bit mono
                for i in range(0, len(pcm), bytes_per_20ms):
                    chunk = pcm[i : i + bytes_per_20ms]
                    if not chunk:
                        break
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
                    # Yield to event loop so Pipecat can pump audio in parallel
                    await asyncio.sleep(0)
            finally:
                yield TTSStoppedFrame()

        async def cleanup(self) -> None:
            await self._client.aclose()
            await super().cleanup()

else:

    class SarvamTTSService:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("pipecat is not installed")
