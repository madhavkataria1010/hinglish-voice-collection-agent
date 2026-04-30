# Test corpus

The 20-minute Hinglish corpus lives here as WAV clips + a JSONL manifest.

## Format

`manifest.jsonl`, one utterance per line:

```json
{"audio": "clip_001.wav", "transcript": "main pachas hazaar de sakta hoon", "language": "hi", "amounts_inr": [50000], "category": "code_switch_amount", "duration_s": 2.7}
```

Required fields:
- `audio` — WAV filename relative to this dir, 16kHz mono PCM preferred.
- `transcript` — gold transcription as a human would write it (Hinglish romanization or Devanagari, both fine).
- `language` — one of `en`, `hi`, `mixed`, `filler`, `noise`.
- `amounts_inr` — list of canonical integer amounts that appear in the utterance, in order. Empty list if none.
- `category` — see `eval/corpus.py` for the enum.
- `duration_s` — clip length in seconds.

## How we built it

We seed-recorded ~150 short clips covering the six categories with a mix of native Hindi and Indian-English speakers (one of each, plus AI-generated fallbacks for coverage gaps). Total runtime ~20 minutes after deduplication. Composition:

| Category              | Clips | Duration |
|-----------------------|-------|----------|
| en_only               |  20   | ~3 min   |
| hi_only               |  25   | ~4 min   |
| code_switch           |  35   | ~5 min   |
| code_switch_amount    |  30   | ~5 min   |
| filler_only           |  25   | ~2 min   |
| noise                 |  15   | ~1 min   |

`code_switch_amount` is the highest-stakes category — every clip has at least one numeric mention spoken across a language flip ("I can pay pachas thousand", "मेरे पास 25 thousand hai").

## Running on a fresh machine

If `manifest.jsonl` is missing, `eval/run_eval.py` exits with a clear error. The corpus is too large to ship in this repo (~80 MB); see the `make corpus` target if a download URL is configured, or generate a placeholder corpus with synthetic TTS via `python -m eval.synth_corpus`.
