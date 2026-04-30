"""Conversation state. The canonical settlement amount lives here, not in transcript text."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Lang = Literal["hi", "en", "mixed"]
OfferKind = Literal["offer", "counteroffer", "settled", "rejected"]


@dataclass
class AmountEvent:
    amount_inr: int
    kind: OfferKind
    speaker: Literal["agent", "borrower"]
    turn_idx: int


@dataclass
class ConversationState:
    principal_inr: int = 50_000
    borrower_name: str = "Rajesh"
    # We start with a name from CRM-style hand-off, but the borrower can
    # correct it ("Actually I'm Aditya"). The LLM is instructed to update it
    # via update_borrower_name; this flag tells the system prompt whether the
    # name is still the unconfirmed default.
    name_confirmed: bool = False

    settlement_amount_inr: int | None = None
    last_borrower_offer_inr: int | None = None
    last_agent_offer_inr: int | None = None

    history: list[AmountEvent] = field(default_factory=list)
    current_lang: Lang = "en"
    lang_confidence: float = 0.5
    turn_idx: int = 0

    def record(self, ev: AmountEvent) -> None:
        self.history.append(ev)
        if ev.speaker == "borrower" and ev.kind in ("offer", "counteroffer"):
            self.last_borrower_offer_inr = ev.amount_inr
        if ev.speaker == "agent" and ev.kind in ("offer", "counteroffer"):
            self.last_agent_offer_inr = ev.amount_inr
        if ev.kind == "settled":
            self.settlement_amount_inr = ev.amount_inr

    def context_for_llm(self) -> str:
        parts = [
            f"Principal owed: ₹{self.principal_inr}",
            f"Borrower name: {self.borrower_name}",
        ]
        if self.last_borrower_offer_inr is not None:
            parts.append(f"Last borrower offer: ₹{self.last_borrower_offer_inr}")
        if self.last_agent_offer_inr is not None:
            parts.append(f"Last agent offer: ₹{self.last_agent_offer_inr}")
        if self.settlement_amount_inr is not None:
            parts.append(f"Settled at: ₹{self.settlement_amount_inr}")
        return "\n".join(parts)
