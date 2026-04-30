# Hinglish Voice Collection Agent

Real-time voice agent for post-default debt collection in India, optimized for the way Indian borrowers actually speak: free code-switching between English and Hindi, often mid-sentence, often mid-number.

Built for the Riverline hiring assignment — Pipecat pipeline, ≥1 OSS component (faster-whisper STT), end-to-end with real mic in / real speaker out, single command to run.

## What this build fixes

The assignment names three failure modes from the existing agent. Each maps to a specific architectural choice in this build:

| Failure mode | Fix in this build | Where in code |
|---|---|---|
| **False language switches** on filler words / noise | Single multilingual TTS voice (no voice flip) + Silero VAD gating + LanguageRouter with word-count thresholds + disfluency dictionary | `nlp/language_router.py`, `nlp/filler.py` |
| **Perceived latency spikes** on language switch | Remote GPU STT (~370ms warm), prefix-cached system prompt, optional backchannel injector, streaming everywhere | `remote_stt/server.py`, `stt/whisper_stt.py`, `nlp/system_prompt.py` |
| **Numeric corruption** across switches | Hinglish/Devanagari/English number normalizer + LLM tool-call (`record_amount`) + canonical-state placeholder substitution + raw-rupee guard | `nlp/number_normalizer.py`, `nlp/turn_processor.py`, `nlp/state.py` |

## Stack

| Layer | Choice | Closed/OSS | Why |
|---|---|---|---|
| Framework | Pipecat 1.1 | OSS | Python-native, first-class `LocalAudioTransport` + `SmallWebRTCTransport` |
| VAD | Silero | OSS | Cheap, accurate; gates STT so noise never reaches Whisper |
| **STT** | **faster-whisper `large-v3`** (CTranslate2) | **OSS** ← **satisfies the assignment's ≥1 OSS rule** | Multilingual, native Hindi/English code-switch in a single forward pass — no separate language detector to flip |
| LLM | OpenAI `gpt-4o` (default; configurable via `OPENAI_MODEL`) — streaming, tool-calling | closed | Lowest TTFT in its class; strong Hinglish instruction-following |
| TTS | Sarvam Bulbul-v2 (Cartesia Sonic-2 fallback) | closed | Most natural Hinglish voice; **single multilingual voice eliminates the 2-3s switch pause at the physical level** |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Browser tab (mic, speaker, native AEC) ◄── WebRTC ──┐                    │
│                                                     │                    │
│ OR LocalAudioTransport (mic, speaker, BotAudioGate) │                    │
└─────────────────────────────────────────────────────┘                    │
                              │                                            │
                              ▼                                            │
       Silero VAD ─► UserTurnProcessor ─► faster-whisper STT ◄─── HTTP ────┘
                                                  │             (SSH tunnel
                                                  │              to A40 GPU)
                                                  ▼
                          number_normalizer + LanguageRouter
                                  │
                                  ▼  user msg = "[REPLY_LANG:hi] <<AMOUNT:50000>> ..."
                          OpenAI LLM (streaming, tools)
                                  │
                                  ▼  "मैं {settlement_amount} पर settle कर सकती हूँ"
                  OutboundTurnProcessor (in-place placeholder substitution)
                                  │
                                  ▼  "मैं 87 हज़ार 500 रुपये पर settle कर सकती हूँ"
                          Sarvam Bulbul-v2 TTS
                                  │
                                  ▼
                          speaker / browser audio
```

State (`ConversationState`) holds the canonical settlement amount as an integer. The transcript and the LLM context never carry rupee figures as free text — they ride as `{settlement_amount}` / `{borrower_offer}` placeholders that get substituted in the language register chosen by the router.

## Quickstart

```bash
git clone https://github.com/madhavkataria1010/hinglish-voice-collection-agent
cd hinglish-voice-collection-agent
cp .env.example .env          # fill in API keys (see below)
uv sync
```

### Required API keys (in `.env`)

| Key | Required for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | LLM | Any Chat-Completions-compatible model id works in `OPENAI_MODEL` |
| `SARVAM_API_KEY` | TTS (recommended) | Falls back to Cartesia if unset |
| `CARTESIA_API_KEY` | TTS (fallback) | Less natural Hinglish |
| `DEEPGRAM_API_KEY` | Eval baseline only | Skip if you only run our pipeline |

### Run modes

There are three ways to run the agent. The browser path is the most reliable on macOS (no Docker mic issues, browser handles AEC).

**1. Browser (recommended on macOS):**
```bash
make tunnel       # opens SSH tunnel to remote GPU STT (skip if WHISPER_BACKEND=local)
make web          # starts FastAPI on http://localhost:7860/
# open http://localhost:7860/ in Chrome, click Connect, talk
```

**2. Local mic / speaker:**
```bash
make tunnel       # remote STT (skip if WHISPER_BACKEND=local)
make run          # talks straight into the laptop mic
```

**3. Docker (single command — works anywhere a browser can reach the host):**
```bash
make docker-build
make docker-run
# open http://localhost:7860/ in Chrome
```

The default `docker compose up` builds the web entry point — no mic
passthrough required, browser handles audio I/O. For the `LocalAudioTransport`
path on a Linux desktop with PulseAudio:

```bash
docker compose -f docker/docker-compose.yml --profile mic up agent-mic
```

> **macOS Docker caveat:** Docker Desktop on macOS cannot pass the host
> microphone into the container reliably. The default web service avoids
> this entirely; use `make web` for the host-Python path if you prefer.

### STT backend choices

`WHISPER_BACKEND` in `.env` selects between:

- **`remote`** (default): `make tunnel` opens an SSH tunnel to a GPU host running `remote_stt/server.py` (faster-whisper `large-v3` on CUDA float16). Warm latency ~370 ms per turn. The remote server is in `remote_stt/` and runs with `./start.sh`.
- **`local`**: runs faster-whisper `large-v3` locally. On Apple Silicon CPU this is ~9–10 s per turn — fine for offline eval, too slow for live demo.

If you don't have a GPU box, switch to `WHISPER_BACKEND=local` for a self-contained run. The eval still works.

## Demo

```bash
# Live conversation (browser is the reliable path)
make web

# Pre-recorded ≥90s demo with three code switches and one mid-number switch
open demo/recording.wav   # to be recorded by the user
```

## Evaluation

```bash
# (1) Generate the test corpus — synthesizes 33 clips via Sarvam TTS.
#     Writes WAVs + manifest.jsonl into eval/test_corpus/
uv run python -m eval.synth_corpus

# (2) Run both pipelines on the corpus and produce eval/report.md
make eval

# Or one-sided runs
make eval-ours
make eval-baseline
```

The report compares our pipeline against a **Deepgram + OpenAI-TTS** baseline on the four target metrics: perceived latency, language detection accuracy, numeric preservation across switches, false-switch rate on non-linguistic audio.

Methodology and caveats are in `eval/metrics.py` (each metric has a top-of-file paragraph explaining how it's measured).

## Project layout

```
.
├── agent.py                       # Local-mic Pipecat pipeline (single command)
├── agent_web.py                   # Browser/WebRTC entry point (FastAPI)
├── stt/whisper_stt.py             # faster-whisper STT — local + remote engines (OSS)
├── tts/sarvam_tts.py              # Sarvam Bulbul-v2 TTS service (Pipecat-compatible)
├── tts/cartesia_fallback.py       # Cartesia Sonic-2 fallback
├── nlp/
│   ├── number_normalizer.py       # Hinglish/Devanagari/English -> canonical int
│   ├── system_prompt.py           # debt-collection prompt + record_amount tool
│   ├── language_router.py         # response-language decision + EN_DEVANAGARI_HINTS
│   ├── filler.py                  # backchannel injector
│   ├── state.py                   # ConversationState (canonical amounts)
│   └── turn_processor.py          # inbound/outbound text rewriting
├── remote_stt/
│   ├── server.py                  # FastAPI Whisper server (runs on GPU host)
│   └── start.sh                   # launches uvicorn with cuDNN paths set
├── eval/
│   ├── corpus.py                  # manifest loader
│   ├── synth_corpus.py            # synthesize corpus via Sarvam
│   ├── baseline_pipeline.py       # Deepgram + OpenAI-TTS reference pipeline
│   ├── metrics.py                 # the four target metrics + methodology
│   └── run_eval.py                # `make eval` entrypoint
├── docker/                        # Dockerfile + docker-compose.yml (default = web; Linux mic via --profile mic)
├── demo/                          # ≥90s recorded demo (recorded manually; not checked in)
├── report/
│   ├── report.tex                 # LaTeX writeup (architecture + measurement)
│   └── report.pdf                 # compiled PDF (committed for reviewers)
├── ARCHITECTURE.md                # markdown architecture writeup
└── DECISION_JOURNAL.md            # handwritten friction log (assignment requires hand-written)
```

## Tests

```bash
make test-normalizer   # 20+ numeric cases must pass
```

`nlp/number_normalizer.py` has a self-test block; `nlp/language_router.py` is tested through the router-driven sanity in the eval.

## Known caveats (honest engineering)

- **macOS Docker can't pass mic.** Use `make web` (browser) or `make run` (host mic) instead.
- **Whisper occasionally mishears proper nouns** in Hindi mode (e.g. "Madhav" → "वादव"). Mitigated by `WHISPER_INITIAL_PROMPT` (the borrower name + collections vocabulary is auto-injected via `BORROWER_NAME` in `.env`); a domain-fine-tune would do better but is out of scope for this submission.
- **Pipecat's `RTVIObserver` has an upstream bug** (the `_bot_transcription` buffer never resets at LLM-turn boundaries). Locally patched at module load in `agent.py`; without the patch each browser-UI turn shows the previous turn's tail prefixed onto the next message.

## Submission deliverables

Mapped to the assignment's submission checklist:

| Required | Where |
|---|---|
| Public GitHub repo with detailed README | this repo + this file |
| Docker Compose setup, single command | `docker/docker-compose.yml` → `make docker-run` |
| Recorded demo (≥90 s, ≥3 code switches, ≥1 mid-number) | `demo/recording.wav` (uploaded with submission) |
| Measurement report with raw data | `eval/report.md` + `eval/results/{ours,baseline}.jsonl` |
| Architecture writeup (1–2 pages) | `ARCHITECTURE.md` (markdown) and `report/report.pdf` (LaTeX, Part I) |
| Decision journal (must be handwritten) | `DECISION_JOURNAL.md` |

Reproduce the eval report from a fresh checkout:
```bash
cp .env.example .env                  # fill keys
uv sync
make tunnel                           # SSH tunnel to remote-STT GPU server (skip if WHISPER_BACKEND=local)
uv run python -m eval.synth_corpus    # synthesize the 34-clip corpus
make eval                             # writes eval/report.md + raw JSONL
```

## License

MIT — see `LICENSE`. © 2026 Madhav Kataria.
