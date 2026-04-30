"""
Hinglish / Devanagari / English numeric normalization.

Converts spoken-number phrases in any of the three forms into canonical INR
integers, so the LLM never has to re-parse "pachas thousand" through its own
tokenizer. Output replaces matched spans with ``<<AMOUNT:50000>>`` markers.

The single source of truth for numeric facts in the agent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #

# Units 0-9 in Hinglish romanization (multiple spellings) + Devanagari
UNITS: dict[str, int] = {
    # English
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    # Hinglish (most common spellings)
    "shunya": 0, "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4,
    "paanch": 5, "panch": 5, "chhe": 6, "chhah": 6, "che": 6,
    "saat": 7, "sat": 7, "aath": 8, "aat": 8, "nau": 9, "no": 9,
    # Devanagari
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9,
}

# Teens 10-19 + tens 20-90 in Hinglish (Hindi has unique forms for many of these)
# Coverage focused on amounts likely to appear in collection negotiations
# (multiples of 5/10 between 5k-100k). We list the common forms.
TEENS_TENS: dict[str, int] = {
    # English teens
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    # English tens
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    # Hinglish tens (most common spellings only — Hindi has a unique
    # word per integer 1-99, but borrowers overwhelmingly use multiples of 10
    # for amounts. Single integers below come through as digits via Whisper.)
    "das": 10, "dus": 10,
    "bees": 20, "bis": 20,
    "tees": 30, "tis": 30,
    "chalis": 40, "chaalis": 40,
    "pachas": 50, "pachaas": 50, "pachhas": 50,
    "saath": 60, "sath": 60,
    "sattar": 70,
    "assi": 80, "asi": 80,
    "nabbe": 90, "navve": 90,
    # Common compound tens that appear as single tokens in casual speech
    "pachees": 25, "pachhees": 25,  # 25
    "pachattar": 75, "pachhattar": 75,  # 75
    # Devanagari (subset)
    "दस": 10, "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50,
    "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
    "पच्चीस": 25, "पचहत्तर": 75,
}

# Scales — Indian system uses lakh (1e5) and crore (1e7); also support
# Western thousand/million for code-switched English speech.
SCALES: dict[str, int] = {
    "hundred": 100, "sau": 100, "सौ": 100,
    "thousand": 1_000, "hazaar": 1_000, "hazar": 1_000,
    "hajaar": 1_000, "hajar": 1_000, "k": 1_000, "हज़ार": 1_000, "हजार": 1_000,
    "lakh": 100_000, "lac": 100_000, "lakhs": 100_000, "lacs": 100_000,
    "लाख": 100_000,
    "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000,
    "करोड़": 10_000_000, "करोड": 10_000_000,
    "million": 1_000_000, "billion": 1_000_000_000,
}

# Tokens we treat as "and" / glue — ignored in number parsing
GLUE = {"and", "aur", "or", "ka", "ke", "ki", "में", "और"}

CURRENCY_TOKENS = {
    "rupees", "rupee", "rs", "rs.", "inr",
    "rupaye", "rupaiya", "rupaiye", "rupiya", "rupye",
    "रुपये", "रुपए", "रुपया", "₹",
}

ALL_NUM_WORDS = set(UNITS) | set(TEENS_TENS) | set(SCALES)


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

# Match word-tokens (letters/Devanagari), digit groups, currency symbol,
# and "₹50,000" / "50k" style. Comma inside digits is preserved as one token.
TOKEN_RE = re.compile(
    r"₹|"                                     # currency symbol
    r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|"        # 50,000 or 1,00,000
    r"\d+(?:\.\d+)?|"                          # plain digits
    r"[A-Za-z]+|"                              # latin words
    r"[ऀ-ॿ]+",                       # Devanagari words
    re.UNICODE,
)


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

@dataclass
class Match:
    start: int
    end: int
    value: int


def _parse_digit_token(tok: str) -> float | None:
    """Parse '50,000' / '50000' / '50.5'. Returns float so '1.5 lakh' scales correctly."""
    s = tok.replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _word_value(tok: str) -> int | None:
    low = tok.lower()
    if low in UNITS:
        return UNITS[low]
    if low in TEENS_TENS:
        return TEENS_TENS[low]
    return None


def _scale_value(tok: str) -> int | None:
    return SCALES.get(tok.lower())


def _try_consume_number(
    toks: list[tuple[str, int, int]], i: int
) -> tuple[Match, int] | None:
    """
    Greedy left-to-right consumption of a numeric phrase starting at index i.
    Handles patterns like:
        "fifty thousand" → 50000
        "pachas hazaar"  → 50000
        "1 lakh 50 thousand" → 150000
        "pachas k" → 50000
        "₹ 50,000" → 50000
        "fifty" → 50
    Currency tokens before/after are absorbed but don't change the value.
    Returns (Match, new_index) or None.
    """
    n = len(toks)
    start_i = i
    # Leading currency symbol (₹) absorbed
    if i < n and toks[i][0] in CURRENCY_TOKENS:
        i += 1

    if i >= n:
        return None

    total: float = 0   # accumulated amount
    current: float = 0 # current "chunk" before next scale
    matched_anything = False
    span_start = toks[i][1]
    span_end = toks[i][2]

    while i < n:
        tok, s, e = toks[i]
        low = tok.lower()

        # Allow glue words inside a number phrase ("one lakh AND fifty thousand")
        if matched_anything and low in GLUE:
            i += 1
            span_end = e
            continue

        # Trailing currency words: "fifty thousand RUPEES" — absorb and stop
        if matched_anything and low in CURRENCY_TOKENS:
            i += 1
            span_end = e
            break

        digit_val = _parse_digit_token(tok)
        if digit_val is not None:
            # If we already have a partial chunk and see a bare digit, that's
            # usually a new number — stop here.
            if current != 0 or total != 0:
                # exception: digit followed by scale ("50 thousand")
                if i + 1 < n and _scale_value(toks[i + 1][0]) is not None:
                    pass  # fall through, will be consumed below
                else:
                    break
            current += digit_val
            matched_anything = True
            i += 1
            span_end = e
            # Look for a scale immediately after
            if i < n:
                scale = _scale_value(toks[i][0])
                if scale is not None:
                    current *= scale
                    total += current
                    current = 0
                    span_end = toks[i][2]
                    i += 1
            continue

        wv = _word_value(tok)
        if wv is not None:
            current += wv
            matched_anything = True
            i += 1
            span_end = e
            continue

        scale = _scale_value(tok)
        if scale is not None:
            if current == 0 and total == 0:
                # bare "thousand" with no leading number — treat as 1×scale
                current = 1
            current *= scale
            total += current
            current = 0
            matched_anything = True
            i += 1
            span_end = e
            continue

        # Not part of a number — stop
        break

    if not matched_anything:
        return None

    total += current
    value = int(round(total))
    if value <= 0:
        return None
    return Match(start=span_start, end=span_end, value=value), i


def find_amounts(text: str) -> list[Match]:
    """Return all numeric-amount spans in *text*, left to right, non-overlapping."""
    toks = _tokenize(text)
    out: list[Match] = []
    i = 0
    while i < len(toks):
        # Skip ahead to a token that could plausibly start a number
        tok = toks[i][0]
        low = tok.lower()
        is_digit = bool(re.match(r"^\d", tok))
        is_currency = low in CURRENCY_TOKENS
        could_start = (
            is_digit
            or is_currency
            or low in UNITS
            or low in TEENS_TENS
            or low in SCALES
        )
        if not could_start:
            i += 1
            continue
        result = _try_consume_number(toks, i)
        if result is None:
            i += 1
            continue
        match, new_i = result
        out.append(match)
        i = new_i
    return out


def normalize(text: str) -> tuple[str, list[int]]:
    """
    Replace every recognized amount phrase with ``<<AMOUNT:N>>`` and return
    (normalized_text, [amounts_in_order]).
    """
    matches = find_amounts(text)
    if not matches:
        return text, []
    out_parts: list[str] = []
    cursor = 0
    amounts: list[int] = []
    for m in matches:
        out_parts.append(text[cursor : m.start])
        out_parts.append(f"<<AMOUNT:{m.value}>>")
        amounts.append(m.value)
        cursor = m.end
    out_parts.append(text[cursor:])
    return "".join(out_parts), amounts


def render_amount(amount_inr: int, lang: str = "en") -> str:
    """
    Render a canonical integer back to a spoken form for TTS.
    For ``lang='hi'`` use Indian numbering (lakh/hazaar). For ``en`` use
    Indian-English ("fifty thousand rupees"). For ``mixed`` default to en
    style — Sarvam Bulbul handles either pronunciation gracefully.
    """
    if amount_inr <= 0:
        return "zero rupees"

    if lang == "hi":
        # Indian system: crore / lakh / hazaar
        crore, rem = divmod(amount_inr, 10_000_000)
        lakh, rem = divmod(rem, 100_000)
        hazaar, rem = divmod(rem, 1_000)
        parts: list[str] = []
        if crore:
            parts.append(f"{crore} crore")
        if lakh:
            parts.append(f"{lakh} lakh")
        if hazaar:
            parts.append(f"{hazaar} hazaar")
        if rem:
            parts.append(str(rem))
        return " ".join(parts) + " rupaye"

    # English (Indian-English with lakh/crore for >=1L for naturalness)
    if amount_inr >= 10_000_000:
        crore, rem = divmod(amount_inr, 10_000_000)
        if rem == 0:
            return f"{crore} crore rupees"
        return f"{crore} crore {rem} rupees"
    if amount_inr >= 100_000:
        lakh, rem = divmod(amount_inr, 100_000)
        thousand = rem // 1_000
        if thousand and rem % 1_000 == 0:
            return f"{lakh} lakh {thousand} thousand rupees"
        if rem == 0:
            return f"{lakh} lakh rupees"
        return f"{lakh} lakh {rem} rupees"
    if amount_inr >= 1_000:
        thousand, rem = divmod(amount_inr, 1_000)
        if rem == 0:
            return f"{thousand} thousand rupees"
        return f"{thousand} thousand {rem} rupees"
    return f"{amount_inr} rupees"


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

_GOLDEN: list[tuple[str, list[int]]] = [
    ("I can pay pachas thousand rupees", [50_000]),
    ("I can pay pachas hazaar rupaye", [50_000]),
    ("main pachas thousand de sakta hoon", [50_000]),
    ("fifty thousand only", [50_000]),
    ("₹50,000 final offer", [50_000]),
    ("50k is what I have", [50_000]),
    ("मैं पचास हज़ार दे सकता हूँ", [50_000]),
    ("one lakh fifty thousand", [150_000]),
    ("ek lakh pachas hazaar", [150_000]),
    ("1.5 lakh", [150_000]),
    ("settle for 35,000 rupees", [35_000]),
    ("pay 25 thousand now and 25 thousand later", [25_000, 25_000]),
    ("haan haan accha", []),
    ("hello are you there", []),
    ("teen lakh", [300_000]),
    ("five hundred", [500]),
    ("paanch sau", [500]),
    ("sirf bees hazaar", [20_000]),
    ("twenty five thousand", [25_000]),
    ("pachees thousand", [25_000]),
]


def _selftest() -> int:
    failed = 0
    for text, expected in _GOLDEN:
        norm, got = normalize(text)
        ok = got == expected
        flag = "OK " if ok else "FAIL"
        print(f"[{flag}] {text!r:55s} -> {got}  (expected {expected})  norm={norm!r}")
        if not ok:
            failed += 1
    print(f"\n{len(_GOLDEN) - failed}/{len(_GOLDEN)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_selftest())
