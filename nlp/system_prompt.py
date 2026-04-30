"""System prompt + tool schema for the debt-collection LLM.

Two design rules enforced by prompt and tool contract:
  1. The LLM never emits a rupee figure as free text. It calls record_amount()
     and references {settlement_amount} / {borrower_offer} placeholders.
  2. The LLM mirrors the borrower's language register (English / Hindi / mixed)
     but does not "switch voices" — that's handled by a single multilingual TTS.
"""
from __future__ import annotations

from .state import ConversationState

SYSTEM_PROMPT = """\
You are Priya, a polite but firm HDFC Bank collections agent. CRM name:
{borrower_name} ({name_confirmation_note}). They defaulted on a
₹{principal_inr} personal loan three months ago. Goal: settle for at
least ₹{min_settlement_inr} (70%) on this call. Replies: 1-2 short
sentences max.

OPEN: First reply is the greeting. One sentence, Indian English, e.g.
"Hello, am I speaking with {borrower_name}? This is Priya from HDFC
Bank about your personal loan." Never produce empty/silent turns.

NAME: If the borrower says a different name or corrects you, call
update_borrower_name with the corrected name and use it from then on.
Otherwise no tool call needed.

LANGUAGE — AUTHORITATIVE TAG:
- Every user message starts with a directive tag: [REPLY_LANG:hi] or
  [REPLY_LANG:en]. Honor it exactly. This tag is computed by an
  external router from word counts (≥5 mostly-Hindi words → hi;
  ≥3 mostly-English words → en) — it is the SOURCE OF TRUTH for what
  language to reply in. Do not second-guess it.
- [REPLY_LANG:hi]  → Reply in Devanagari-script Hindi (देवनागरी), not
  Latin transliteration. The TTS picks the Hindi voice from the script.
- [REPLY_LANG:en]  → Reply in clean Indian English.
- Never echo the tag in your reply.

NUMBERS — CRITICAL:
- Never write rupee amounts as digits or words in reply text.
- When an amount comes up, call record_amount with a canonical integer
  (no commas, no currency), then reference {{settlement_amount}} or
  {{borrower_offer}} as a literal placeholder in your reply. The system
  substitutes the canonical figure.
- User transcripts pre-tag numbers as <<AMOUNT:N>>. Trust the tag.

TONE: Empathetic, professional, never threatening. No legal threats.
If the borrower offers below ₹{min_settlement_inr}, counter higher;
at-or-above, accept and confirm.

State (offers, settlement) is inferred from conversation history and
your own tool calls — there is no separate state block.
"""


def build_tools_schema():
    """Build the Pipecat ToolsSchema for record_amount.

    Imported lazily so the module loads cleanly without pipecat installed
    (e.g. for unit tests of the pure-python pieces).
    """
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    record_amount = FunctionSchema(
        name="record_amount",
        description=(
            "Record any rupee amount mentioned in the conversation. Call this "
            "BEFORE writing your reply text whenever a number comes up. The "
            "amount must be a canonical integer (no commas, no currency)."
        ),
        properties={
            "amount_inr": {
                "type": "integer",
                "minimum": 1,
                "description": "Amount in rupees as an integer.",
            },
            "kind": {
                "type": "string",
                "enum": ["offer", "counteroffer", "settled", "rejected"],
                "description": (
                    "offer: agent's proposed amount; "
                    "counteroffer: borrower's proposal; "
                    "settled: both parties agreed; "
                    "rejected: explicitly declined."
                ),
            },
            "speaker": {
                "type": "string",
                "enum": ["agent", "borrower"],
                "description": "Who proposed this amount.",
            },
        },
        required=["amount_inr", "kind", "speaker"],
    )

    update_borrower_name = FunctionSchema(
        name="update_borrower_name",
        description=(
            "Call this when the borrower corrects their name (e.g. they say "
            "'actually I'm Aditya' or 'no, I'm Priya not Rajesh'). The new "
            "name will be used in all subsequent replies. Pass exactly what "
            "the borrower said their name is, with normal capitalisation."
        ),
        properties={
            "name": {
                "type": "string",
                "description": "Borrower's confirmed name.",
            },
        },
        required=["name"],
    )

    return ToolsSchema(standard_tools=[record_amount, update_borrower_name])


def build_system_prompt(state: ConversationState) -> str:
    """Render the system prompt for the current call.

    Per-call substitutions only (borrower_name, principal). The mid-call
    state block was removed — the LLM infers state from conversation
    history + its own tool calls. Keeping the system prompt stable
    across turns lets OpenAI prompt-cache the (system + early history)
    prefix once it grows past the cache threshold.
    """
    if state.name_confirmed:
        note = "the borrower has confirmed this is correct"
    else:
        note = (
            "this has NOT been confirmed yet — it may be wrong, the borrower "
            "may go by a different name, or the call may have reached someone "
            "else entirely"
        )
    return SYSTEM_PROMPT.format(
        borrower_name=state.borrower_name,
        name_confirmation_note=note,
        principal_inr=state.principal_inr,
        min_settlement_inr=int(state.principal_inr * 0.70),
    )
