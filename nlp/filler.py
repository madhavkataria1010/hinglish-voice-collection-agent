"""
Backchannel / filler injector — the perceived-latency trick.

The instant the borrower stops speaking (VAD EOU), we emit a 100-200ms
acknowledgement audio clip ("haan", "achha", "ji", "samjha") to the speaker
while the LLM is still composing. By the time the real response audio starts
streaming from TTS, the borrower has already heard us, so they don't perceive
a pause.

Fillers are language-aware: we pick a Hindi filler if the last user utterance
was Hindi-leaning, English ("right", "got it") if English-leaning. We also
deduplicate so we don't emit two of the same filler in a row, which would
sound robotic.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Literal

Lang = Literal["hi", "en", "mixed"]

FILLERS: dict[Lang, list[str]] = {
    "hi": ["haan ji", "achha", "samjha", "ji", "theek hai", "hmm"],
    "en": ["right", "got it", "okay", "I see", "mm-hmm", "sure"],
    "mixed": ["haan ji", "achha", "right", "okay", "samjha"],
}

# Conditions under which we suppress the filler — emitting one would feel wrong:
#  - very short user turn (<500ms): borrower might still be speaking
#  - back-to-back fillers from agent: stop sounding robotic
#  - user turn that's itself a single filler ("hello?")


class FillerInjector:
    def __init__(
        self,
        enabled: bool = True,
        cooldown_s: float = 1.5,
        min_user_turn_s: float = 0.5,
        history_size: int = 3,
    ) -> None:
        self.enabled = enabled
        self.cooldown_s = cooldown_s
        self.min_user_turn_s = min_user_turn_s
        self._recent: deque[str] = deque(maxlen=history_size)
        self._last_emit_ts: float = 0.0

    def maybe_pick(
        self,
        lang: Lang,
        user_turn_duration_s: float,
        now_ts: float,
        user_text: str,
    ) -> str | None:
        if not self.enabled:
            return None
        if user_turn_duration_s < self.min_user_turn_s:
            return None
        if now_ts - self._last_emit_ts < self.cooldown_s:
            return None
        # Don't ack if the user's whole turn was filler-only — they're waiting
        words = user_text.strip().split()
        if len(words) <= 1:
            return None

        candidates = [f for f in FILLERS[lang] if f not in self._recent]
        if not candidates:
            candidates = FILLERS[lang]
        choice = random.choice(candidates)
        self._recent.append(choice)
        self._last_emit_ts = now_ts
        return choice
