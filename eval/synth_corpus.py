"""
Generate a *synthetic* test corpus when no recorded one is available.

Uses Sarvam TTS to render gold transcripts to WAV, so the eval pipeline can
run end-to-end on a fresh checkout. This is a fallback for development /
reproducibility — real evaluation should use human recordings.

Run:
    python -m eval.synth_corpus
"""
from __future__ import annotations

import asyncio
import json
import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Curated 60+ utterances spanning all six categories.
# Each tuple: (transcript, language, amounts_inr, category, target_lang_code)
SEED: list[tuple[str, str, list[int], str, str]] = [
    # en_only
    ("Hello, this is regarding my loan.", "en", [], "en_only", "en-IN"),
    ("I cannot pay the full amount right now.", "en", [], "en_only", "en-IN"),
    ("Please tell me the settlement option.", "en", [], "en_only", "en-IN"),
    ("I lost my job last month.", "en", [], "en_only", "en-IN"),
    ("Can you give me one week to arrange?", "en", [], "en_only", "en-IN"),
    ("I am willing to settle this today.", "en", [], "en_only", "en-IN"),
    # hi_only
    ("main abhi paise nahi de sakta", "hi", [], "hi_only", "hi-IN"),
    ("mujhe thoda time chahiye", "hi", [], "hi_only", "hi-IN"),
    ("aap kya offer kar rahe ho", "hi", [], "hi_only", "hi-IN"),
    ("ghar mein paisa nahi hai bhai", "hi", [], "hi_only", "hi-IN"),
    ("kya main agle mahine de sakta hoon", "hi", [], "hi_only", "hi-IN"),
    # code_switch
    ("haan main settlement ke liye ready hoon", "mixed", [], "code_switch", "hi-IN"),
    ("please thoda samay dijiye", "mixed", [], "code_switch", "hi-IN"),
    ("I will try but mujhe time lagega", "mixed", [], "code_switch", "hi-IN"),
    ("okay madam batao kitna dena hai", "mixed", [], "code_switch", "hi-IN"),
    ("EMI miss ho gayi sorry", "mixed", [], "code_switch", "hi-IN"),
    # code_switch_amount — the headline target
    ("I can pay pachas thousand rupees", "mixed", [50_000], "code_switch_amount", "hi-IN"),
    ("main pachas thousand de sakta hoon", "mixed", [50_000], "code_switch_amount", "hi-IN"),
    ("settle for thirty five thousand please", "en", [35_000], "code_switch_amount", "en-IN"),
    ("paintees hazaar mein settle kar lo", "hi", [35_000], "code_switch_amount", "hi-IN"),
    ("twenty five thousand abhi aur baaki next month", "mixed", [25_000], "code_switch_amount", "hi-IN"),
    ("ek lakh bahut zyada hai forty thousand chalega", "mixed", [100_000, 40_000], "code_switch_amount", "hi-IN"),
    ("I will pay 50k today", "en", [50_000], "code_switch_amount", "en-IN"),
    ("chalis hazaar final hai sir", "hi", [40_000], "code_switch_amount", "hi-IN"),
    ("thirty thousand please that's all I have", "en", [30_000], "code_switch_amount", "en-IN"),
    ("tees hazaar de dunga abhi", "hi", [30_000], "code_switch_amount", "hi-IN"),
    # filler_only — these MUST NOT trigger language flip
    ("haan", "filler", [], "filler_only", "hi-IN"),
    ("hmm", "filler", [], "filler_only", "hi-IN"),
    ("achha", "filler", [], "filler_only", "hi-IN"),
    ("okay", "filler", [], "filler_only", "en-IN"),
    ("yes", "filler", [], "filler_only", "en-IN"),
    ("ji", "filler", [], "filler_only", "hi-IN"),
    # noise — silence/ambient. We synthesize empty-ish clips for these.
    ("", "noise", [], "noise", "en-IN"),
    ("", "noise", [], "noise", "en-IN"),
]


async def _synthesize_one(client, transcript: str, target_lang: str) -> bytes:
    if not transcript:
        # 0.5s of silence at 22.05kHz, 16-bit mono
        return b"\x00\x00" * int(22_050 * 0.5)
    pcm, _ttfb = await client.synthesize(transcript, target_language_code=target_lang)
    return pcm


def _write_wav(path: Path, pcm: bytes, sr: int = 22_050) -> float:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return len(pcm) / 2 / sr


async def main() -> None:
    if not os.getenv("SARVAM_API_KEY"):
        raise SystemExit("SARVAM_API_KEY not set; cannot synthesize corpus.")
    from tts.sarvam_tts import SarvamClient

    out_dir = Path("eval/test_corpus")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    client = SarvamClient()
    try:
        with manifest_path.open("w") as mf:
            for idx, (text, lang, amounts, cat, target_lang) in enumerate(SEED, 1):
                fname = f"clip_{idx:03d}.wav"
                logger.info(f"[{idx}/{len(SEED)}] {cat:20s} {text[:50]!r}")
                pcm = await _synthesize_one(client, text, target_lang)
                duration = _write_wav(out_dir / fname, pcm)
                mf.write(
                    json.dumps(
                        {
                            "audio": fname,
                            "transcript": text,
                            "language": lang,
                            "amounts_inr": amounts,
                            "category": cat,
                            "duration_s": round(duration, 2),
                        }
                    )
                    + "\n"
                )
    finally:
        await client.aclose()
    logger.info(f"Wrote {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
