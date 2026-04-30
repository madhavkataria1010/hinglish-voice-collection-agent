"""
faster-whisper STT — the OSS slot in our pipeline.

Why Whisper for Hinglish:
  - Multilingual model handles Hindi+English in a single forward pass without
    a separate language detector flipping the pipeline. This is the key reason
    we don't see the "false language switch" failure mode at the STT layer.
  - large-v3 (Nov 2023) is materially better on code-switched Hinglish than
    v2/medium in our testing (see DECISION_JOURNAL).
  - faster-whisper (CTranslate2) gives us int8 quantization for usable
    latency on Apple Silicon CPU/Metal — we measure ~200-350ms per 1-second
    audio chunk on M2.

We wrap it as a Pipecat ``STTService`` so it slots cleanly into the pipeline.
We also expose a stateless ``transcribe(audio)`` for the eval harness.
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import time
import wave
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

try:
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        Frame,
        TranscriptionFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.services.stt_service import SegmentedSTTService
    from pipecat.transcriptions.language import Language
    PIPECAT_AVAILABLE = True
except Exception:  # pragma: no cover - pipecat optional for unit tests
    PIPECAT_AVAILABLE = False
    SegmentedSTTService = object  # type: ignore[misc,assignment]
    Language = None  # type: ignore[assignment]


@dataclass
class WhisperResult:
    text: str
    language: str
    avg_logprob: float
    no_speech_prob: float
    duration_s: float
    latency_s: float


class FasterWhisperEngine:
    """Stateless wrapper around the faster-whisper model. Lazy-loaded."""

    _instance: "FasterWhisperEngine | None" = None

    def __init__(
        self,
        model_size: str = "large-v3",
        compute_type: str = "int8",
        device: str = "auto",
    ) -> None:
        from faster_whisper import WhisperModel

        if device == "auto":
            # Apple Silicon: faster-whisper uses CPU with int8; CoreML builds
            # are out of scope for this build. CUDA selected automatically on
            # Linux+NVIDIA.
            try:
                import torch  # noqa: F401

                device = "cuda" if _cuda_available() else "cpu"
            except Exception:
                device = "cpu"

        logger.info(
            f"Loading faster-whisper model={model_size} compute={compute_type} "
            f"device={device}"
        )
        t0 = time.time()
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info(f"Whisper loaded in {time.time() - t0:.2f}s")
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device

    @classmethod
    def shared(cls) -> "FasterWhisperEngine":
        if cls._instance is None:
            cls._instance = cls(
                model_size=os.getenv("WHISPER_MODEL", "large-v3-turbo"),
                compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
                device=os.getenv("WHISPER_DEVICE", "auto"),
            )
        return cls._instance

    def transcribe_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
        language: str | None = None,
    ) -> WhisperResult:
        """
        Transcribe a mono float32 numpy array.

        Language behaviour: defaults to ``WHISPER_LANGUAGE`` env (set to "hi"
        for Hinglish in practice). We initially tried language=None to let
        Whisper auto-detect, but on real Hinglish input it picked Urdu about
        half the time and emitted Arabic-script transcripts that corrupted
        names through the LLM. Pinning to Hindi keeps it in Devanagari/Latin
        and Whisper still handles English code-switch inline within the same
        forward pass — exactly what we need for the false-switch target.

        condition_on_previous_text=False is set because long-context
        conditioning compounds errors across turns (well-known Whisper issue),
        which directly hurts the numeric-preservation target.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if sample_rate != 16_000:
            raise ValueError("Whisper requires 16kHz mono audio")

        if language is None:
            env_lang = os.getenv("WHISPER_LANGUAGE", "hi").strip()
            language = env_lang or None

        duration = len(audio) / sample_rate
        t0 = time.time()
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=1,                  # greedy = lower latency
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,             # we VAD upstream in Pipecat
            without_timestamps=True,
            word_timestamps=False,
        )
        segs = list(segments)
        text = "".join(s.text for s in segs).strip()
        latency = time.time() - t0
        avg_lp = (
            float(np.mean([s.avg_logprob for s in segs])) if segs else -1.0
        )
        no_speech = (
            float(np.mean([s.no_speech_prob for s in segs])) if segs else 1.0
        )
        return WhisperResult(
            text=text,
            language=info.language or "unknown",
            avg_logprob=avg_lp,
            no_speech_prob=no_speech,
            duration_s=duration,
            latency_s=latency,
        )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class RemoteWhisperEngine:
    """HTTP client for the remote-STT service in ``remote_stt/server.py``.

    Mac CPU int8 large-v3 was running at ~9-10s per turn. The same model on
    an A40 in float16 is ~50-150ms. We POST the raw WAV (already in hand
    from SegmentedSTTService) and get back the same WhisperResult shape as
    the local engine.
    """

    def __init__(
        self,
        url: str,
        language: str | None = None,
        timeout_s: float = 30.0,
        initial_prompt: str | None = None,
    ) -> None:
        import httpx

        self.url = url.rstrip("/")
        # Match local engine: env-driven, default 'hi' for Hinglish.
        if language is None:
            env_lang = os.getenv("WHISPER_LANGUAGE", "hi").strip()
            language = env_lang or None
        self.language = language
        # Soft vocabulary bias: passed to Whisper's decoder to make proper
        # nouns and domain words more likely. Without this, names like
        # "Madhav" came back as "वादव" because Hindi-mode Whisper has no
        # acoustic pull toward English proper nouns.
        if initial_prompt is None:
            initial_prompt = os.getenv("WHISPER_INITIAL_PROMPT") or None
        self.initial_prompt = initial_prompt
        self._http = httpx.Client(timeout=timeout_s)
        logger.info(
            f"RemoteWhisperEngine -> {self.url} "
            f"(lang={self.language!r}, prompt={'set' if initial_prompt else 'none'})"
        )

    def transcribe_wav_bytes(self, wav: bytes) -> WhisperResult:
        params: dict[str, str] = {}
        if self.language:
            params["language"] = self.language
        if self.initial_prompt:
            params["initial_prompt"] = self.initial_prompt
        t0 = time.time()
        resp = self._http.post(
            f"{self.url}/transcribe",
            content=wav,
            params=params,
            headers={"Content-Type": "application/octet-stream"},
        )
        wall = time.time() - t0
        if resp.status_code != 200:
            raise RuntimeError(f"remote STT {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        # latency_s from the server is GPU inference only; we log it as such
        # but caller sees the network round-trip via its own clock if it cares.
        return WhisperResult(
            text=data.get("text", ""),
            language=data.get("language", "unknown"),
            avg_logprob=float(data.get("avg_logprob", -1.0)),
            no_speech_prob=float(data.get("no_speech_prob", 1.0)),
            duration_s=float(data.get("duration_s", 0.0)),
            latency_s=float(data.get("latency_s", wall)),
        )


# Whisper's most common silence-triggered hallucinations across languages.
# These come from its YouTube training corpus (subtitle credits, music tags,
# stock end-of-video phrases). Drop them when they appear with high confidence
# but no actual speech — they are not what the borrower said.
_HALLUCINATION_PATTERNS = [
    "thanks for watching",
    "thank you for watching",
    "subscribe",
    "subtitles by",
    "subtitled by",
    "transcription by",
    "teksting av",          # Norwegian — what the user just hit
    "untertitel von",       # German
    "sous-titres",          # French
    "sottotitoli",          # Italian
    "subtítulos",           # Spanish
    "amara.org",
    "captions by",
    "[music]",
    "[applause]",
    "[silence]",
    "♪",
    "you you you",
    "bye bye bye",
    "मेरी आवाज़",
]


_ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def _is_known_hallucination(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    # Hard rejection of Arabic-script output. We pin WHISPER_LANGUAGE=hi
    # specifically to keep the model from falling into Urdu — if it ever
    # does anyway (rare, but observed on noisy/short clips), we drop the
    # turn rather than send Urdu to the LLM and have it propagate into
    # the borrower name or amounts. There is no Hindi conversation that
    # legitimately contains Arabic-script characters.
    if _ARABIC_SCRIPT_RE.search(text):
        return True
    for pat in _HALLUCINATION_PATTERNS:
        if pat in low:
            return True
    # Whole-utterance repetition: same token >=4 times in a row.
    toks = low.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return True
    # Run-length repetition anywhere in the transcript: any single token
    # repeating >=6 times consecutively. Catches the faster-whisper
    # degenerate-loop failure mode like
    #   "अगर आप इस प्रति प्रति प्रति प्रति प्रति प्रति प्रति ..."
    # where the prefix is real but the model loops on one token. We can't
    # safely keep the prefix because the model's hidden state was poisoned
    # by the loop — better to drop the whole turn.
    if len(toks) >= 6:
        run = 1
        for i in range(1, len(toks)):
            if toks[i] == toks[i - 1]:
                run += 1
                if run >= 6:
                    return True
            else:
                run = 1
    return False


# --------------------------------------------------------------------------- #
# Pipecat integration
# --------------------------------------------------------------------------- #

if PIPECAT_AVAILABLE:

    class WhisperSTTService(SegmentedSTTService):  # type: ignore[misc]
        """
        Pipecat STT service. ``SegmentedSTTService`` buffers audio between
        VADUserStartedSpeakingFrame and VADUserStoppedSpeakingFrame, then
        hands us a complete WAV (header + PCM) per user turn — exactly one
        run_stt call per utterance, never per 20ms frame.

        Running Whisper on the whole segment is materially more accurate than
        sliding-window for the short turns we see in this scenario, and it
        kills the per-frame hallucination loop we got from STTService.
        """

        def __init__(
            self,
            engine: FasterWhisperEngine | RemoteWhisperEngine | None = None,
            bot_echo_grace_s: float = 0.5,
            **kwargs: Any,
        ) -> None:
            from pipecat.services.settings import STTSettings

            kwargs.setdefault("sample_rate", 16_000)
            # Tell Pipecat we don't pin a model name (we configure faster-whisper
            # ourselves) and that language is None because Whisper auto-detects.
            kwargs.setdefault(
                "settings", STTSettings(model="faster-whisper-large-v3", language=None)
            )
            super().__init__(**kwargs)
            if engine is None:
                backend = os.getenv("WHISPER_BACKEND", "local").lower()
                # Build a default initial_prompt that biases the decoder
                # toward the borrower's name + collections vocabulary. This
                # is what stops "Madhav" being heard as "वादव" / "बादब".
                # We bias both English (the actual name) and a Devanagari
                # rendering, plus rupee/settlement words. Sticking strictly
                # to vocabulary the user is likely to say keeps the prompt
                # short — long prompts hurt latency.
                borrower_name = os.getenv("BORROWER_NAME", "").strip()
                default_prompt_parts = [
                    p for p in [
                        borrower_name,
                        "HDFC Bank, Priya, settlement, payment, loan, EMI, "
                        "rupees, hazaar, lakh, UPI, account, balance.",
                    ] if p
                ]
                default_prompt = " ".join(default_prompt_parts) if default_prompt_parts else None
                if backend == "remote":
                    url = os.getenv("WHISPER_REMOTE_URL", "http://localhost:8765")
                    engine = RemoteWhisperEngine(url=url, initial_prompt=default_prompt)
                else:
                    engine = FasterWhisperEngine.shared()
            self._engine = engine
            self._is_remote = isinstance(engine, RemoteWhisperEngine)
            # Echo defense: a transcript that completes while (or just after)
            # the bot was speaking is bot's own audio leaking through the mic.
            # We discard it. Whisper hallucinates Hindi credit-rolls on this
            # leakage even with no_speech filters, so this is the load-bearing
            # check for "is this real user speech."
            self._bot_speaking = False
            self._bot_stopped_ts: float = 0.0
            self._bot_echo_grace_s = bot_echo_grace_s

        async def process_frame(
            self, frame: Frame, direction: FrameDirection
        ) -> None:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_speaking = False
                self._bot_stopped_ts = time.time()
            await super().process_frame(frame, direction)

        async def run_stt(
            self, audio: bytes
        ) -> AsyncGenerator[Frame, None]:
            # Gate 0: drop the whole turn if the bot was speaking (or just
            # stopped). The mic almost certainly captured TTS bleed-through.
            if self._bot_speaking or (
                time.time() - self._bot_stopped_ts < self._bot_echo_grace_s
            ):
                logger.debug("STT skip: bot was speaking (echo guard)")
                return
            try:
                with wave.open(io.BytesIO(audio), "rb") as wf:
                    sr = wf.getframerate()
                    nframes = wf.getnframes()
                    raw = wf.readframes(nframes)
            except Exception as e:
                logger.warning(f"STT skip: WAV decode failed ({e})")
                return

            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if len(arr) == 0:
                return

            if sr != 16_000:
                # Cheap linear resample. Pipecat is configured for 16 kHz so
                # this is a safety net, not the hot path.
                ratio = 16_000 / sr
                idx = (np.arange(int(len(arr) * ratio)) / ratio).astype(np.int64)
                idx = np.clip(idx, 0, len(arr) - 1)
                arr = arr[idx]

            # 1) Energy gate — reject sub-silence chunks before Whisper sees them.
            #    -45 dBFS is conservative for desktop mic; ambient room noise
            #    typically sits at -55 to -50 dBFS.
            rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
            rms_dbfs = 20.0 * np.log10(max(rms, 1e-9))
            if rms_dbfs < -45.0:
                logger.debug(f"STT skip: silence ({rms_dbfs:.1f} dBFS)")
                return
            if len(arr) < 0.4 * 16_000:  # <400ms — too short to be a real turn
                logger.debug(f"STT skip: short chunk ({len(arr)/16_000:.2f}s)")
                return

            loop = asyncio.get_event_loop()
            if self._is_remote:
                # Send the original WAV bytes — server decodes them itself.
                result = await loop.run_in_executor(
                    None,
                    lambda: self._engine.transcribe_wav_bytes(audio),  # type: ignore[union-attr]
                )
            else:
                result = await loop.run_in_executor(
                    None,
                    lambda: self._engine.transcribe_array(arr, sample_rate=16_000),  # type: ignore[union-attr]
                )

            text = (result.text or "").strip()
            if not text:
                return

            # 2) Whisper confidence filters. no_speech_prob > 0.5 catches the
            #    Hindi/multilingual hallucinations Whisper emits on near-silent
            #    audio. We saw real speech come through at 0.05-0.30, so 0.5
            #    is a safe cut.
            if result.no_speech_prob > 0.5:
                logger.debug(
                    f"STT skip: no_speech_prob={result.no_speech_prob:.2f} "
                    f"text={text!r}"
                )
                return
            if result.avg_logprob < -1.0:
                logger.debug(
                    f"STT skip: low logprob={result.avg_logprob:.2f} text={text!r}"
                )
                return

            # 3) Known-hallucination blocklist. These appear in Whisper's
            #    training data as caption credits and surface on silence.
            if _is_known_hallucination(text):
                logger.debug(f"STT skip: hallucination {text!r}")
                return

            logger.info(
                f"STT [{result.language}] ({result.latency_s*1000:.0f}ms, "
                f"lp={result.avg_logprob:.2f}, ns={result.no_speech_prob:.2f}, "
                f"rms={rms_dbfs:.1f}dB): {text}"
            )
            lang_enum = None
            if Language is not None and result.language:
                code = result.language.split("-")[0].lower()
                lang_enum = getattr(Language, code.upper(), None)
            yield TranscriptionFrame(
                text=text,
                user_id="borrower",
                timestamp=str(time.time()),
                language=lang_enum,
            )

else:

    class WhisperSTTService:  # type: ignore[no-redef]
        """Stub when Pipecat isn't installed (eval-only contexts)."""

        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError(
                "pipecat is not installed; use FasterWhisperEngine directly "
                "for offline transcription."
            )
