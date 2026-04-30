"""
Language-of-response decision with hysteresis.

We don't change the TTS *voice* on language switch (single multilingual voice).
This module only decides what register the LLM should reply in, and feeds the
filler injector. Hysteresis prevents single-token Hindi inserts (or noise that
slipped past VAD) from flipping our state.

Heuristic, not a model:
  1. Tokenize the latest user transcript.
  2. Count Devanagari tokens, common Hinglish-romanization tokens, and English
     stopwords. Compute hindi_ratio = (hi_tokens) / (hi + en).
  3. Apply hysteresis: only flip state if hindi_ratio is past the opposite
     threshold for two consecutive turns OR is very strong this turn (>0.8).
  4. Single-token utterances and pure-filler utterances never flip state.
"""
from __future__ import annotations

import re
from typing import Literal

Lang = Literal["hi", "en", "mixed"]

# Tokens that are pure filler / disfluency — never count toward language ratio
DISFLUENCY = {
    "haan", "han", "haa", "hmm", "mm", "uh", "um", "umm", "uhh", "arre",
    "achha", "accha", "ok", "okay", "ji", "yeah", "yes", "no", "na",
    "हाँ", "अच्छा", "जी",
}

# Strong English-only signal tokens (these almost never appear in Hindi speech)
EN_HINTS = {
    "the", "and", "is", "are", "was", "were", "have", "has", "had", "will",
    "would", "should", "could", "this", "that", "these", "those", "what",
    "when", "where", "why", "how", "okay", "please", "thank", "thanks",
    "rupees", "rupee", "amount", "settle", "settlement", "loan", "payment",
    "pay", "money", "thousand", "hundred", "lakh",  # neutral/Indian
}

# Strong Hindi-only Romanization signal tokens
HI_HINTS = {
    "main", "mein", "mai", "tum", "aap", "hum", "mera", "tera", "uska",
    "hoon", "hain", "tha", "thi", "the", "kya", "kyun", "kaise", "kab",
    "kahan", "kaha", "yahan", "wahan", "abhi", "aaj", "kal", "phir",
    "kar", "karo", "karna", "ho", "hoga", "hogi", "hain", "hai",
    "rupaye", "rupiya", "paisa", "paise", "hazaar", "hajaar",
    "bhai", "yaar", "dost", "ghar", "kaam", "naukri", "paisa",
    "nahi", "nahin", "nai", "haan", "ji",
    "saa", "se", "ko", "ki", "ka", "ke", "par", "mein", "tak",
    "pachas", "bees", "tees", "saath",
}

# English words rendered in Devanagari by Whisper. Because we pin
# WHISPER_LANGUAGE=hi (to avoid Urdu mistranscription), pure English
# speech also comes back as Devanagari — e.g. "I can pay you around
# 90,000" -> "आई केंड पे यू अराउंड 90,000". Without this set, the router
# would only see Devanagari and conclude the user is speaking Hindi,
# even when they're speaking English. Each entry here is a Devanagari
# transliteration of a common English word; matching it counts as an
# English signal, NOT a Hindi one.
#
# Whisper-Hindi-mode is inconsistent about how it transliterates English
# (नो vs नौ, थैंक्स vs थैंक्यू, ओके vs ओक्के), so we cast a wide net of
# common variants. False positives are limited because almost none of
# these collide with real Hindi words at this register.
EN_DEVANAGARI_HINTS = {
    # ---- pronouns ----
    "आई", "यू", "वी", "ही", "शी", "इट", "दे", "देम",
    "मी", "अस", "हिम", "हर", "हिज", "हर्स", "इट्स",
    "माय", "माईन", "यॉर", "यॉर्स", "अवर", "आवर", "देयर",
    "दिस", "दैट", "दीज़", "दोज़",
    # ---- aux / modals ----
    "एम", "इज़", "इज", "आर", "वाज़", "वाज", "वर", "बीन", "बीइंग", "बी",
    "हैव", "हैज", "हैड", "हैविंग",
    "डू", "डिड", "डन", "डूइंग", "डज",
    "विल", "वुड", "शुड", "कुड", "मस्ट", "मे", "माइट",
    "कैन", "कैंट", "केंट", "केंड",
    "डोंट", "डिडंट", "विलनॉट", "वोंट",
    # ---- common verbs (base + ing + past) ----
    "स्पीक", "स्पीकिंग", "स्पोक", "स्पोकन",
    "टॉक", "टॉकिंग", "टॉक्ड",
    "से", "सेइंग", "सेड", "सेज",
    "कॉल", "कॉलिंग", "कॉल्ड",
    "गेट", "गॉट", "गोट", "गेटिंग",
    "गिव", "गेव", "गिविंग", "गिवन",
    "टेक", "टुक", "टेकिंग", "टेकन",
    "मेक", "मेड", "मेकिंग",
    "गो", "वेंट", "गोइंग", "गॉन",
    "कम", "केम", "कमिंग",
    "पे", "पेय", "पेइंग", "पेड",
    "वांट", "वांटेड", "वांटिंग",
    "नीड", "नीडेड", "नीडिंग",
    "लाइक", "लव", "लाइकिंग",
    "नो", "न्यू", "नोइंग", "नोन",
    "थिंक", "थॉट", "थिंकिंग",
    "मीन", "मेंट", "मीनिंग",
    "ट्राय", "ट्राईड", "ट्राइंग",
    "स्टार्ट", "स्टार्टेड", "स्टार्टिंग",
    "फिनिश", "फिनिश्ड",
    "हेल्प", "हेल्प्ड", "हेल्पिंग",
    "हियर", "हर्ड", "हियरिंग",
    "सी", "सॉ", "सीइंग", "सीन",
    "वर्क", "वर्किंग", "वर्क्ड",
    "वेट", "वेटिंग", "वेटेड",
    "सेटल", "सेटलिंग", "सेटल्ड",
    # ---- prepositions / conjunctions ----
    "टू", "ऑफ", "इन", "ऑन", "एट", "बाय", "फॉर", "विद",
    "विदाउट", "विदिन", "फ्रॉम", "इंटू", "ऑन्टू", "अपटू",
    "आउट", "अप", "डाउन", "ओवर", "अंडर",
    "अराउंड", "अबाउट", "अगेंस्ट", "बिटवीन", "अमंग",
    "आफ्टर", "बिफोर", "ड्यूरिंग", "थ्रू", "थ्रूआउट",
    "बट", "एंड", "ऑर", "नॉर", "सो", "बीकॉज़", "इफ", "देन", "एल्स",
    # ---- question / interjection ----
    "वाट", "व्हाट", "वेन", "व्हेन", "वेयर", "व्हेयर",
    "वाई", "व्हाई", "हाउ", "हू", "हूम", "विच",
    "यस", "येस", "ये", "येप",
    "ओके", "ओक्के", "ओकिडोक",
    "हाय", "हेलो", "हैलो", "हाई", "बाय", "बायबाय",
    "प्लीज़", "प्लीज", "थैंक्स", "थैंक", "थैंक्यू", "सॉरी",
    # ---- articles / quantifiers (most common ones get pure-Devanagari forms)
    "अ", "एन", "द", "सम", "एनी", "नो", "ऑल", "बोथ", "इच",
    "मच", "मेनी", "फेव", "मोर", "मोस्ट", "लेस", "लीस्ट",
    # ---- adverbs / time ----
    "नाउ", "देन", "हीयर", "देयर",
    "टुडे", "टुमॉरो", "येस्टरडे", "टुनाइट",
    "सून", "लेटर", "लेट", "अर्ली",
    "ऑलवेज", "नेवर", "सम्टाइम्स", "ऑफन", "यूज़ुअली",
    "मेबी", "रियली", "वेरी", "जस्ट", "ऑनली", "ऑल्सो", "टू",
    # ---- common adjectives ----
    "गुड", "बैड", "बेटर", "बेस्ट", "वर्स", "वर्स्ट",
    "हाई", "लो", "बिग", "स्मॉल", "हैप्पी", "हैपी",
    "फाइन", "राइट", "रॉन्ग", "श्योर", "ट्रू", "फॉल्स",
    # ---- domain: banking / money ----
    "बैंक", "बैंकिंग", "लोन", "लोन्स", "इएमआई", "ईएमआई",
    "सेटल", "सेटलमेंट", "अमाउंट", "अमाउंट्स",
    "पेमेंट", "पेमेंट्स", "इन्स्टॉलमेंट", "इंस्टॉलमेंट",
    "इंट्रेस्ट", "इन्ट्रेस्ट", "इंटरेस्ट",
    "क्रेडिट", "डेबिट", "अकाउंट", "अकाउन्ट",
    "बैलेंस", "ड्यू", "ड्यूज़", "ओवरड्यू",
    "मनी", "कैश", "चेक", "ट्रांसफर", "ट्रांज़ैक्शन",
    "रुपीज़", "रुपीस", "रुपीज", "रुपी", "रुपीज़,",
    "डॉलर", "डॉलर्स",
    "थाउज़ंड", "थाउज़ेंड", "थाउसैंड", "थाउसेंड",
    "हंड्रेड", "हंडरेड", "मिलियन", "बिलियन",
    # ---- English number words (in Devanagari) ----
    "वन", "टू", "थ्री", "फोर", "फाईव", "फाइव",
    "सिक्स", "सेवन", "एट", "नाइन", "टेन",
    "इलेवन", "ट्वेल्व", "थर्टीन", "फोर्टीन", "फिफ्टीन",
    "सिक्स्टीन", "सेवेंटीन", "एटीन", "नाइनटीन",
    "ट्वेंटी", "थर्टी", "फोर्टी", "फिफ्टी",
    "सिक्स्टी", "सेवेंटी", "एटी", "नाइनटी",
    # ---- common verbs the borrower might use ----
    "अंडरस्टैंड", "अग्री", "एक्सेप्ट", "रिजेक्ट", "कन्फर्म",
    "प्रॉमिस", "ट्राय", "एक्सप्लेन", "डिस्कस", "रिक्वेस्ट",
    "कंसीडर", "कांसिडर", "अरेंज", "मैनेज", "अफोर्ड",
}

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def _classify_token(tok: str) -> Lang | None:
    low = tok.lower()
    if low in DISFLUENCY:
        return None
    # Devanagari that is actually transliterated English (e.g. "आई", "पे",
    # "अराउंड") must be classified as English, otherwise pinned-to-Hindi
    # Whisper output would force every English utterance to look Hindi.
    # Check this BEFORE the script-based Devanagari->Hindi rule.
    if tok in EN_DEVANAGARI_HINTS:
        return "en"
    if DEVANAGARI_RE.search(tok):
        return "hi"
    if low in HI_HINTS:
        return "hi"
    if low in EN_HINTS:
        return "en"
    return None


def hindi_ratio(text: str) -> tuple[float, int]:
    """Return (ratio, total_signal_tokens). Ratio is 0..1, hindi share."""
    toks = re.findall(r"[A-Za-z]+|[ऀ-ॿ]+", text)
    hi = en = 0
    for t in toks:
        c = _classify_token(t)
        if c == "hi":
            hi += 1
        elif c == "en":
            en += 1
    total = hi + en
    if total == 0:
        return 0.5, 0
    return hi / total, total


class LanguageRouter:
    """
    Maintains current_lang with hysteresis. Call ``observe(user_text)`` after
    each user turn; read ``current`` to decide LLM reply register.

    Concrete rules (the user's explicit asks):
      - A turn with >=5 signal tokens that is mostly (>=80%) Hindi flips
        the router to ``hi`` immediately, no streak required.
      - A turn with >=3 signal tokens that is mostly (<=30% Hindi)
        English flips back to ``en`` immediately.
      - Mid-confidence (somewhere in between) only flips after two
        consecutive same-direction turns. This prevents a single
        confused transcript from oscillating us.
      - Single-word utterances and pure fillers ("haan", "ji", "okay")
        never flip; they hold the current state.
    """

    def __init__(self, initial: Lang = "en") -> None:
        self.current: Lang = initial
        self.confidence: float = 0.5
        # Need two consecutive opposing-side turns to flip, unless ratio extreme
        self._streak_lang: Lang | None = None
        self._streak_count: int = 0

    def observe(self, user_text: str) -> Lang:
        ratio, signal = hindi_ratio(user_text)
        # Too few signal tokens (e.g. user said "okay" or single-word filler):
        # do NOT flip. Just hold the current language.
        if signal < 2:
            return self.current

        # User-requested deterministic flip rules (no hysteresis needed):
        #   - 5+ tokens AND >=80% Hindi -> definitely Hindi.
        #   - 3+ tokens AND <=30% Hindi -> definitely English.
        if signal >= 5 and ratio >= 0.80:
            self.current = "hi"
            self.confidence = ratio
            self._streak_lang = "hi"
            self._streak_count = 1
            return self.current
        if signal >= 3 and ratio <= 0.30:
            self.current = "en"
            self.confidence = 1 - ratio
            self._streak_lang = "en"
            self._streak_count = 1
            return self.current

        # Otherwise compute a tentative label
        if ratio >= 0.6:
            new = "hi"
        elif ratio <= 0.4:
            new = "en"
        else:
            new = "mixed"

        # Mid-confidence: require two-turn streak before flipping
        if new == self._streak_lang:
            self._streak_count += 1
        else:
            self._streak_lang = new
            self._streak_count = 1

        if self._streak_count >= 2 and new != self.current:
            self.current = new
            self.confidence = max(ratio, 1 - ratio)
        return self.current

    def reply_lang(self) -> Literal["hi", "en"]:
        """Return the language the LLM should reply in.

        Collapses the internal "mixed" state to a binary decision so we
        can give the LLM a clear directive: either Devanagari Hindi or
        Indian English. Mid-mixed turns inherit whatever the previous
        turn was — biased toward stability.
        """
        if self.current == "hi":
            return "hi"
        if self.current == "en":
            return "en"
        # mixed: lean toward whichever the streak is biased to, default en
        return "hi" if self._streak_lang == "hi" else "en"
