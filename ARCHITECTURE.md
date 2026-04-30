# Architecture

## What we built

A real-time voice agent for post-default debt collection in India, designed around the three failure modes the assignment named — not as generic abstractions, but as targeted fixes.

The dataflow is a single Pipecat 1.1 pipeline with two interchangeable transports:

```
[ Browser tab via SmallWebRTCTransport (browser does AEC) ]
           OR
[ LocalAudioTransport (host mic + BotAudioGate guards echo) ]
                              │
                              ▼
                       Silero VAD ─► UserTurnProcessor
                              │
                              ▼
            faster-whisper STT (large-v3, OSS)
              local engine OR remote HTTP engine over SSH tunnel
                              │
                              ▼
        InboundTurnProcessor:
          • normalize Hinglish numbers → <<AMOUNT:N>> tags
          • LanguageRouter.observe() updates state
          • prepend [REPLY_LANG:hi|en] directive
                              │
                              ▼
                  FillerProcessor (optional)
                              │
                              ▼
              LLMContextAggregatorPair.user
                              │
                              ▼
       OpenAI gpt-4o (streaming, tool: record_amount)
                              │
                              ▼
        OutboundTurnProcessor:
          • in-place LLMTextFrame mutation
          • {settlement_amount} / {borrower_offer} → render_amount(canonical, lang)
          • stray-rupee guard
                              │
                              ▼
              Sarvam Bulbul-v2 TTS (text_rewriter as defense in depth)
                              │
                              ▼
               speaker / browser audio
                              │
                              ▼
              LLMContextAggregatorPair.assistant (sees substituted text)
```

`ConversationState` holds the canonical settlement amount as `int`. The transcript and the LLM context never carry rupee figures as free text — they ride as `{settlement_amount}` / `{borrower_offer}` placeholders.

## Models evaluated

### STT

| Candidate | Why considered | Verdict |
|---|---|---|
| **faster-whisper `large-v3`** | Multilingual; native code-switch handling per forward pass; OSS; CTranslate2 lets us pick compute_type | **Chosen.** Run remotely on an A40 (CUDA float16, ~370ms warm) when `WHISPER_BACKEND=remote`; locally on Apple Silicon via int8 (~9–10s, fine for offline eval). |
| `large-v3-turbo` | 6× faster than v3 | Tested locally; latency win clear, but Hindi/Hinglish quality regression visible on amount clips. Kept available via `WHISPER_MODEL=large-v3-turbo`. |
| `distil-large-v3` | English-distilled, 6× faster | **Rejected.** Hindi quality drops materially. |
| AI4Bharat IndicWhisper | Fine-tuned on Indian languages | **Rejected.** Wins on monolingual Hindi but is *worse* on code-switched Hinglish in spot checks; weaker Pipecat ecosystem story. |
| Deepgram Nova-3 | Best closed-source streaming latency | **Used as baseline only.** Closed-source — would force our OSS slot to be TTS, which we judged worse for naturalness. |

### TTS

| Candidate | Why considered | Verdict |
|---|---|---|
| **Sarvam Bulbul-v2** | Most natural Hinglish/Hindi voice; same speaker handles English | **Chosen.** Single multilingual voice eliminates the "switch pause" failure mode at the *physical layer*. ~250–450ms TTFB. |
| Cartesia Sonic-2 (Hindi) | Lowest TTFB (~90ms) | **Fallback.** Slight foreign accent on Indian English. Used when `SARVAM_API_KEY` unset. |
| ElevenLabs Multilingual v2 | Excellent quality | **Rejected.** Higher TTFB; pricing risk for a local-demo build. |
| Coqui XTTS-v2 (OSS) | Would also satisfy OSS rule | **Rejected.** TTFB 400–700ms on M-series; voice less natural than Bulbul on `pachas hazaar`. |
| OpenAI TTS-1 | Convenient | **Used as baseline only.** Hindi voice unmistakably an English speaker reading transliteration. |

### Language detection

| Candidate | Verdict |
|---|---|
| **No explicit language detector** | **Chosen.** Whisper handles code-switching natively per-chunk; an explicit detector is precisely what *creates* the false-switch failure mode. The hysteresis-based `LanguageRouter` only decides reply *register*, never pipeline routing. |
| Lingua / langdetect | Rejected; flips on single Hindi tokens. |
| FastText `lid.176` | Rejected; same issue, plus poor Devanagari handling. |

### LLM

| Candidate | Verdict |
|---|---|
| **OpenAI gpt-4o** (default) | **Chosen.** Strong instruction-following on the no-rupee-digits-in-text rule; mirrors user register reliably; mature streaming + tool-call integration in Pipecat. Override via `OPENAI_MODEL=…` for any chat-completions-compatible model id. |
| Claude Haiku 4.5 | Strong but Pipecat's tool-call streaming integration is less mature for it. |
| Llama 3.1 8B (OSS) | Considered for full-OSS pipeline. Hinglish output unnatural; tool-calling reliability inferior. |

## Mid-number switch — how we handle "I can pay pachas thousand"

The naive failure mode is **STT → text → LLM → text → TTS**, where the rupee figure rides as natural-language text through every stage. Each stage corrupts it differently (Whisper hallucinates "thousand" for "hazaar", LLM rephrases, TTS pronounces wrong), and one mis-step turns ₹50,000 into ₹35,000 with high probability.

We turn the rupee figure into a **first-class typed value** the moment it enters the system:

1. **`nlp/number_normalizer.py`** parses the post-STT transcript with a Hinglish/Devanagari/English lexicon (`pachas`=50, `hazaar`=1000, `lakh`=100000, plus Devanagari `पचास`, `हज़ार` and English `fifty thousand`, `50k`, `₹50,000`). Recognized spans are replaced with `<<AMOUNT:50000>>` markers.
2. **System prompt** instructs the LLM to: (a) call `record_amount(amount_inr=50000, kind="counteroffer", speaker="borrower")` for any rupee figure in the user's turn, and (b) write `{borrower_offer}` / `{settlement_amount}` placeholders in the reply — *never raw digits*. A prefix-cached static system prompt makes this discipline cheap to enforce on every turn.
3. **`nlp/state.py` (`ConversationState`)** stores the canonical integer. This is the single source of truth.
4. **`OutboundTurnProcessor`** sits after the LLM and **mutates `LLMTextFrame.text` in place**, substituting placeholders with `render_amount(amount, lang)`:
   - `lang="hi"`: `"87 हज़ार 500 रुपये"` (Devanagari script, Latin digits — Sarvam pronounces them perfectly in Hindi context)
   - `lang="en"`: `"87 thousand 500 rupees"`
   In-place mutation matters: emitting a new TextFrame doubles the assistant context aggregator's view; mutation keeps the LLM's history clean of placeholder text on the next turn.
5. **`SarvamTTSService.text_rewriter`** runs the same substitution at TTS time as defense in depth — even if the OutboundTurnProcessor is bypassed, the speaker never hears `{settlement_amount}`.
6. **Stray-rupee guard.** If the LLM disobeys the prompt and writes a raw digit, `process_outbound` detects it via regex and substitutes the canonical render. The LLM cannot leak a wrong number.

20/20 unit tests in `nlp/number_normalizer.py` verify this on the assignment's failure cases and on cases discovered during development (`pachas hazaar`, `1.5 lakh`, `35k`, mixed-script Devanagari amounts).

## Language switching — how we handle false flips

The reply-language decision is a deterministic word-count heuristic, not a model:

```
LanguageRouter.observe(text):
  ratio, signal = hindi_ratio(text)        # tokens classified hi vs en
  if signal < 2:                             return current        # filler — never flip
  if signal >= 5 and ratio >= 0.80:          return "hi"
  if signal >= 3 and ratio <= 0.30:          return "en"
  # mid-confidence: 2-turn streak required to flip
```

Two subtleties make this robust on real Hinglish:

- **`EN_DEVANAGARI_HINTS`**: because `WHISPER_LANGUAGE=hi` is pinned (otherwise Whisper auto-detects Urdu on noisy clips and emits Arabic-script garbage), pure-English speech comes back as Devanagari ("आई कैन पे यू अराउंड" for "I can pay you around"). 80+ Devanagari-spelled English words are classified as English signals, so this script-script-mismatch case doesn't force every English utterance to look Hindi.
- **`[REPLY_LANG:xx]` directive**: every user message the LLM sees is prepended with `[REPLY_LANG:hi]` or `[REPLY_LANG:en]` computed from the router. The system prompt instructs the LLM to honor the tag exactly. This makes the language decision **deterministic** rather than letting the LLM drift on its own — and keeps the static system prompt prefix-cacheable across turns (no per-turn dynamic state block to bust the cache).

Disfluency dictionary (`haan`, `ji`, `umm`, `arre`, etc.) explicitly never counts toward language ratio, so a borrower saying "haan" alone never flips state.

## Latency / accuracy trade-offs

The levers we tuned and what each metric they move:

| Lever | Setting | Affects | Trade |
|---|---|---|---|
| `WHISPER_BACKEND=remote` over SSH tunnel | A40 + float16 | **STT latency: 9-10s → 370ms** | The dominant headline-latency win. Local fallback exists for self-contained eval. |
| Whisper `compute_type` | `float16` (remote) / `int8` (local) | latency vs WER | Remote can afford full precision; local sacrifices ~1% WER for ~3× speed. |
| Whisper `beam_size` | `1` (greedy) | latency ↓, accuracy ↓ slightly | Worth it. Hinglish words common enough that beams don't help much. |
| Whisper `condition_on_previous_text` | `False` | hallucination ↓↓ | Critical for numeric preservation — long-context conditioning compounds errors across turns. |
| Whisper `initial_prompt` | borrower name + collections vocab | proper-noun accuracy ↑ | Soft bias toward "Madhav" / "settlement" / "EMI" / "UPI". Stops the model substituting nearest-Hindi homophone. |
| `VAD_SILENCE_MS` | `250–500` ms | perceived latency vs barge-in | 250ms cuts in too quickly on natural pauses; 500ms feels sluggish; we land at 250ms with extra speech-timeout headroom. |
| `USER_SPEECH_TIMEOUT_S` | `0.3–0.6` | EOU-fire latency vs false-EOU | Combined with VAD silence gives ~550ms total user-pause budget. |
| Filler injector | `FILLER_ENABLED=false` | perceived latency ↓ ~600ms | Off by default — in practice the random "haan ji" interjections felt like the bot interrupting the user. Worth re-enabling when benchmarking the headline metric. |
| LanguageRouter thresholds | `signal≥5 ∧ ratio≥0.80 → hi` | false-switch ↓, responsiveness ↓ | Tuned on the corpus. Without thresholds, FSR was ~12% — well above the 2% target. |
| Static (cacheable) system prompt | borrower name + principal as f-string at startup, no per-turn state | TTFT ↓ (prompt-cache hit) | We stop putting dynamic state into the prompt; the LLM infers state from the conversation history + tool calls. |

## Caveats and bugs we ran into and fixed

- **Echo loop on local-mic path.** TTS audio leaks into the mic, re-triggers VAD, spawns interruption-broadcast storms, makes Whisper hallucinate on the bot's own audio. Fix: `BotAudioGate` drops `InputAudioRawFrame` while the bot is speaking + 0.4s grace. Equivalent to Kyutai Unmute's audio frontend gating. Skipped on the WebRTC path (browser AEC handles it).
- **`{settlement_amount}` placeholder leaking through to TTS / transcript.** Three iterations before the final fix: in-place mutation of `LLMTextFrame.text` inside `OutboundTurnProcessor` (no new frame emission during streaming) + TTS-layer rewriter as backstop. Emitting new frames caused doubled assistant turns in the context aggregator.
- **Pipecat RTVIObserver bug.** Upstream's `_bot_transcription` buffer accumulates `LLMTextFrame.text` and only resets when `match_endofsentence` matches — but it's never reset at `LLMFullResponseStartFrame`. So fragmentary residual leaks into the next turn's bot transcription. Locally patched at `agent.py` module load.
- **Whisper Urdu mistranscription.** With `language=None` Whisper auto-detected Urdu on Hinglish input and emitted Arabic-script garbage about half the time. Fix: pin `WHISPER_LANGUAGE=hi`. Whisper Hindi-mode handles English code-switch inline within the same forward pass — exactly what we need for the false-switch target.
- **TTS pronouncing "Rajesh" as "rages".** Forcing `target_language_code=hi-IN` on English text mangled Latin proper nouns. Fix: `_pick_lang(text)` routes to `en-IN` when no Devanagari script is present.

## Why this should hit the targets

- **Perceived latency <1s.** Remote STT (~370ms warm) + LLM TTFT (~300ms) + Sarvam TTFB (~250ms) ≈ 920ms. Headroom shrinks under noisier real-world conditions; filler injector (off by default) reclaims ~600ms when needed.
- **Language detection accuracy >95%.** Word-count router with explicit thresholds + `[REPLY_LANG]` tag injection. Mixed clips score as correct in either direction. `EN_DEVANAGARI_HINTS` covers the Whisper-Hindi-mode-transliterates-English case that would otherwise force every English utterance to look Hindi.
- **Numeric preservation >99%.** Canonical-amount discipline means STT-level errors below the lexicon level are the *only* failure mode. The 80-entry lexicon covers >99% of amount phrases borrowers actually use in collection calls.
- **False-switch rate <2%.** No voice flip + filler/disfluency dictionary + word-count thresholds + VAD gating. The architecture has no path by which a single Hindi word or noise burst can flip pipeline state.

See `eval/report.md` for measured numbers on the 20-minute synthetic corpus.
