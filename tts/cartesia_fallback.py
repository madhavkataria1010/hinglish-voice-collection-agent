"""
Cartesia Sonic-2 TTS — fallback when SARVAM_API_KEY isn't available.

Cartesia has lower TTFB than Sarvam but the Hindi voice is less natural for
borrowers (lighter accent, less colloquial). We use it only when Sarvam is
unreachable so the demo still runs.
"""
from __future__ import annotations

import os

try:
    from pipecat.services.cartesia.tts import CartesiaTTSService
except Exception:  # pragma: no cover
    CartesiaTTSService = None  # type: ignore[assignment]


# A multilingual Sonic voice id known to handle Hindi+English well.
# Override via env var if Cartesia rotates ids.
DEFAULT_CARTESIA_VOICE = os.getenv(
    "CARTESIA_VOICE_ID", "f9836c6e-a0bd-460e-9d3c-f7299fa60f94"
)


def make_cartesia_tts(api_key: str | None = None, voice: str | None = None):
    if CartesiaTTSService is None:
        raise RuntimeError(
            "pipecat.services.cartesia is not available. Install pipecat-ai[cartesia]."
        )
    api_key = api_key or os.getenv("CARTESIA_API_KEY")
    if not api_key:
        raise RuntimeError("CARTESIA_API_KEY not set")
    return CartesiaTTSService(
        api_key=api_key,
        voice_id=voice or DEFAULT_CARTESIA_VOICE,
        model="sonic-2",
        params=CartesiaTTSService.InputParams(language="hi", speed="normal"),
    )
