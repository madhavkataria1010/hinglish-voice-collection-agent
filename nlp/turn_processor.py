"""
Turn-aware text rewriting frames sit between STT and the LLM, and between the
LLM and TTS. Two responsibilities:

  1. Inbound (STT -> LLM): tag amounts in the user's transcript with
     <<AMOUNT:N>> markers, update the language router, log timestamps.
  2. Outbound (LLM -> TTS): substitute {settlement_amount} / {borrower_offer}
     placeholders with rendered amounts in the chosen language. The LLM never
     emits raw rupee figures (per system prompt); if it does anyway, we strip
     them — the canonical-state amount is authoritative.

The whole point of these two passes is the ">99% numeric preservation" target.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from loguru import logger

from .language_router import LanguageRouter
from .number_normalizer import normalize, render_amount
from .state import AmountEvent, ConversationState

PLACEHOLDER_RE = re.compile(
    r"\{(settlement_amount|borrower_offer|agent_offer|principal)\}"
)
# Match a digit-form rupee figure left in the LLM output despite the prompt:
#   "35000", "35,000", "₹50,000", "rs 25000", "1.5 lakh", "50k"
RAW_RUPEE_RE = re.compile(
    r"(?:₹\s*|rs\.?\s*|inr\s*)?"
    r"\d+(?:[,]\d+)*(?:\.\d+)?"
    r"(?:\s*(?:rupees?|rupaye|rupiya|hazaar|hazar|hajar|thousand|lakh|crore|k))?",
    re.IGNORECASE,
)


@dataclass
class InboundResult:
    normalized_text: str
    detected_amounts: list[int]
    language: str


def process_inbound(
    raw_text: str,
    state: ConversationState,
    router: LanguageRouter,
) -> InboundResult:
    """Tag amounts and update conversation state. Returns the normalized text
    that we send to the LLM."""
    state.turn_idx += 1
    normalized, amounts = normalize(raw_text)
    lang = router.observe(raw_text)
    state.current_lang = lang  # type: ignore[assignment]
    state.lang_confidence = router.confidence

    for amt in amounts:
        state.record(
            AmountEvent(
                amount_inr=amt,
                kind="counteroffer",  # speaker=borrower, default to counteroffer
                speaker="borrower",
                turn_idx=state.turn_idx,
            )
        )

    if amounts:
        logger.info(f"Inbound amounts: {amounts}  lang={lang}")

    return InboundResult(
        normalized_text=normalized, detected_amounts=amounts, language=lang
    )


def process_outbound(
    llm_text: str,
    state: ConversationState,
) -> str:
    """Substitute placeholders and strip stray rupee figures.

    Order matters: we run the raw-rupee guard FIRST (on the LLM's original
    text), then placeholder substitution. If placeholders were used correctly
    the guard sees no digits; if the LLM disobeyed and wrote a raw figure,
    we replace it with the canonical render *before* substitution, so the
    canonical render is never re-matched as "raw" digits.
    """
    lang = state.current_lang

    canonical = (
        state.settlement_amount_inr
        or state.last_agent_offer_inr
        or state.last_borrower_offer_inr
    )
    text = llm_text
    if canonical is not None and _has_raw_rupee_figure(text):
        logger.warning(
            "LLM emitted raw rupee figure; substituting canonical amount"
        )
        # Replace ONLY the first match; subsequent digits (e.g. dates) left alone
        text = RAW_RUPEE_RE.sub(
            lambda _m: render_amount(canonical, lang=lang), text, count=1
        )

    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        amt: int | None
        if key == "settlement_amount":
            amt = state.settlement_amount_inr or state.last_agent_offer_inr
        elif key == "borrower_offer":
            amt = state.last_borrower_offer_inr
        elif key == "agent_offer":
            amt = state.last_agent_offer_inr
        elif key == "principal":
            amt = state.principal_inr
        else:
            amt = None
        if amt is None:
            logger.warning(f"Placeholder {{{key}}} unresolved")
            return "the agreed amount"
        return render_amount(amt, lang=lang)

    return PLACEHOLDER_RE.sub(_sub, text)


def _has_raw_rupee_figure(text: str) -> bool:
    """Detect a digit sequence that looks like a rupee amount.

    We look for runs of >=3 digits OR a digit followed by a scale word —
    enough to catch "35000", "35,000", "1.5 lakh", "50k" without firing
    on small ordinals like "1 week" or "in 2 days".
    """
    if re.search(r"\d{3,}", text):
        return True
    if re.search(
        r"\d+\s*(?:rupees?|rupaye|hazaar|thousand|lakh|crore|k)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    return False


@dataclass
class TurnLatency:
    eou_ts: float
    first_llm_token_ts: float | None = None
    first_tts_byte_ts: float | None = None
    first_audio_out_ts: float | None = None

    def perceived_latency_ms(self) -> float | None:
        if self.first_audio_out_ts is None:
            return None
        return (self.first_audio_out_ts - self.eou_ts) * 1000.0

    @classmethod
    def now(cls) -> "TurnLatency":
        return cls(eou_ts=time.time())
