# Hinglish Voice Collection Agent

Real-time voice agent for post-default debt collection on a ₹50,000 personal loan, optimized for the way Indian borrowers actually speak: free code-switching between English and Hindi, often mid-sentence, often mid-number.

Three concrete failure modes the assignment calls out, and what we did about each:

| Failure mode | Fix in this build |
|---|---|
| **False language switches** on filler words / noise | Single multilingual TTS voice (no voice flip) + Silero VAD gating + LanguageRouter hysteresis |
| **Perceived latency spikes** on language switch | Backchannel injector emits "haan ji" / "achha" at EOU while the LLM composes — first audio out in <300ms |
| **Numeric corruption** across switches | Hinglish/Devanagari/English number normalizer + LLM tool-call (`record_amount`) + canonical-state placeholder substitution |

## Stack

| Layer | Choice | Closed/OSS |
|---|---|---|
| Framework | Pipecat | OSS |
| VAD | Silero | OSS |
| **STT** | **faster-whisper `large-v3` int8** | **OSS** ← satisfies the assignment's ≥1 OSS rule |
| LLM | OpenAI `gpt-4o-mini` (streaming, tool calls) | closed |
| TTS | Sarvam Bulbul-v2 (Cartesia Sonic-2 fallback) | closed |

## Quickstart

### macOS / Apple Silicon (recommended dev path)

```bash
cp .env.example .env
# fill in OPENAI_API_KEY and SARVAM_API_KEY (or CARTESIA_API_KEY)

uv sync
make run
```

The first run downloads the Whisper `large-v3` model (~3GB) into the HuggingFace cache. Subsequent runs are instant.

### Linux + Docker (the "we will run it on our machine" path)

```bash
cp .env.example .env
make docker-build
make docker-run
```

> **macOS Docker caveat:** Docker Desktop on macOS cannot pass the host microphone into the container reliably. On a Mac, use `make run` instead. The compose file works on Linux desktops with PulseAudio.

## Demo

```bash
# Live conversation, mic in / speaker out
make run

# Pre-recorded 90s demo with three code switches and one mid-number switch
open demo/recording.wav
```

## Evaluation

```bash
# Generate a synthetic 60-clip corpus (if you don't have a recorded one)
uv run python -m eval.synth_corpus

# Run both pipelines and produce eval/report.md
make eval
```

The report compares our pipeline against a Deepgram + OpenAI-TTS baseline on the four target metrics.

## Project layout

```
.
├── agent.py                 # Pipecat pipeline entrypoint (live agent)
├── stt/whisper_stt.py       # faster-whisper STT (OSS)
├── tts/sarvam_tts.py        # Sarvam Bulbul-v2 TTS
├── tts/cartesia_fallback.py
├── nlp/
│   ├── number_normalizer.py # Hinglish/Devanagari/English -> canonical int
│   ├── system_prompt.py     # debt-collection prompt + record_amount tool
│   ├── language_router.py   # response-language decision with hysteresis
│   ├── filler.py            # backchannel injector
│   ├── state.py             # ConversationState (canonical amounts)
│   └── turn_processor.py    # in/out frame rewriting
├── eval/
│   ├── corpus.py            # manifest loader
│   ├── synth_corpus.py      # generate synthetic corpus via Sarvam
│   ├── baseline_pipeline.py # Deepgram + OpenAI TTS for comparison
│   ├── metrics.py           # the four target metrics + methodology
│   └── run_eval.py          # `make eval` entrypoint
├── docker/                  # Dockerfile + docker-compose.yml
├── demo/                    # ≥90s recorded demo (gitignored)
├── ARCHITECTURE.md
└── DECISION_JOURNAL.md      # hand-written friction log (mandatory deliverable)
```

## Tests

```bash
make test-normalizer   # 20/20 numeric cases must pass
```

## License

Single-author submission for the Riverline hiring assignment.
