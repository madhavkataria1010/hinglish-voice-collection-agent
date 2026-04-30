"""
Test-corpus loader.

The corpus is a directory of WAV clips paired with a JSONL manifest. Each
manifest line is a single utterance:

    {
      "audio": "clip_001.wav",
      "transcript": "main pachas hazaar de sakta hoon",
      "language": "hi",                        # 'hi' | 'en' | 'mixed' | 'noise'
      "amounts_inr": [50000],                  # canonical integers, in order
      "category": "code_switch_amount",        # see CATEGORIES
      "duration_s": 2.7
    }

Categories drive metric breakdown:
    en_only           - pure English, baseline sanity
    hi_only           - pure Hindi/Hinglish, code-switch capability
    code_switch       - mid-sentence English<->Hindi flip
    code_switch_amount- a number is uttered across a code switch
    filler_only       - "haan", "hmm", noise — false-switch test
    noise             - ambient noise, no speech
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = (
    "en_only",
    "hi_only",
    "code_switch",
    "code_switch_amount",
    "filler_only",
    "noise",
)


@dataclass
class CorpusItem:
    audio_path: Path
    transcript: str
    language: str
    amounts_inr: list[int]
    category: str
    duration_s: float


def load_corpus(corpus_dir: str | Path) -> list[CorpusItem]:
    root = Path(corpus_dir)
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Corpus manifest not found at {manifest}. "
            "See eval/test_corpus/README.md for format."
        )
    items: list[CorpusItem] = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        items.append(
            CorpusItem(
                audio_path=root / rec["audio"],
                transcript=rec["transcript"],
                language=rec["language"],
                amounts_inr=rec.get("amounts_inr", []),
                category=rec.get("category", "en_only"),
                duration_s=rec["duration_s"],
            )
        )
    return items
