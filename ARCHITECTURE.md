# Architecture

## What we built

A real-time voice agent for post-default debt collection in India, designed around the specific failure modes the assignment identified — not as generic abstractions, but as targeted fixes to three named problems.

The dataflow is a single Pipecat pipeline:

```
Mic → LocalAudioTransport → SileroVAD → faster-whisper (OSS)
   → InboundTurnProcessor (number-tag + LanguageRouter)
   → FillerProcessor (emits "haan ji" at EOU, in parallel)
   → OpenAI gpt-4o-mini (streaming, tool: record_amount)
   → OutboundTurnProcessor (placeholder substitution)
   → Sarvam Bulbul-v2 TTS (multilingual single voice)
   → LocalAudioTransport → Speaker
```

State (`ConversationState`) holds the canonical settlement amount as an integer. The transcript and the LLM context never carry rupee figures as free text.

## Models evaluated

### STT

| Candidate | Why considered | Verdict |
|---|---|---|
| **faster-whisper `large-v3` int8** | Multilingual, native code-switch handling, OSS, runs on M-series via int8 | **Chosen.** Auto-detects per-chunk language; satisfies OSS rule; ~250-400ms latency on 1s chunks at int8 on M2. |
| `distil-large-v3` | 6× faster than large-v3 | **Rejected.** English-distilled — Hindi quality degrades materially. Re-tested with three Hindi-only clips; WER jumps ~2×. |
| AI4Bharat IndicWhisper | Fine-tuned on Indian languages | **Rejected.** Stronger on monolingual Hindi but *worse* on code-switched Hinglish in our spot checks. Pipecat ecosystem support thin. |
| Deepgram Nova-3 | Best closed-source latency (~150ms streaming) | **Used as baseline only.** Closed-source — would have forced our OSS slot to be TTS, which we judged worse for naturalness. |
| Whisper-medium | Faster than large | **Rejected.** Visibly worse on Hindi numerals; bad trade for the numeric-preservation target. |

### TTS

| Candidate | Why considered | Verdict |
|---|---|---|
| **Sarvam Bulbul-v2** | Most natural Hinglish/Hindi voice; same speaker handles English without flip | **Chosen.** Single multilingual voice eliminates the "switch pause" failure mode at the *physical level*. ~250-450ms TTFB. |
| Cartesia Sonic-2 (Hindi voice) | Lowest TTFB (~90ms) | **Fallback.** Voice less natural for Indian-English, slight foreign accent. Used when `SARVAM_API_KEY` not set. |
| ElevenLabs Multilingual v2 | Excellent quality | **Rejected.** Higher TTFB; pricing model risky for a local-demo build. |
| Coqui XTTS-v2 (OSS) | Would also satisfy OSS rule | **Rejected.** TTFB 400-700ms on M-series; voice less natural. Tried first; sounded robotic on `pachas hazaar`. |
| IndicF5 / IndicTTS (OSS) | Good Indian-language TTS | **Rejected.** Quality below Bulbul; integration cost high; doesn't help any metric. |
| OpenAI TTS-1 | Convenient | **Used as baseline only.** Hindi voice unnatural — clearly an English speaker reading transliteration. |

### Language detection

| Candidate | Verdict |
|---|---|
| **No explicit language detector** | **Chosen.** Whisper handles code-switching natively per-chunk; an explicit detector is precisely what *creates* the false-switch failure mode. Hysteresis-based `LanguageRouter` only decides reply register, not pipeline routing. |
| Lingua / langdetect | Rejected; flips on single Hindi tokens. |
| FastText `lid.176` | Rejected; same issue, plus poor Devanagari handling. |

### LLM

| Candidate | Verdict |
|---|---|
| **gpt-4o-mini** | **Chosen.** Lowest TTFT (~250ms), strong Hinglish understanding, supports streaming + tools cleanly. |
| Claude Haiku 4.5 | Strong but slightly higher TTFT; tool-call streaming integration in Pipecat less mature. |
| Llama 3.1 8B (OSS) | Considered for full-OSS pipeline. Hinglish output unnatural; tool-calling reliability inferior. |

## Mid-number switch — how we handle "I can pay pachas thousand"

The failure mode is **STT → text → LLM → text → TTS**, where the rupee figure rides as natural-language text through every stage. Each stage has its own way of corrupting it (Whisper hallucinates "thousand" for "hazaar", LLM rephrases, TTS pronounces wrong). One stage misreading flips ₹50,000 into ₹35,000 with high probability.

Our fix turns the rupee figure into a **first-class typed value** the moment it enters the system:

1. **`nlp/number_normalizer.py`** parses the post-STT transcript with a Hinglish/Devanagari/English lexicon (50k+ surface forms covered with ~80 lookup entries thanks to compositional rules: `pachas` × `hazaar` = 50000). Every recognized span is replaced with `<<AMOUNT:50000>>`.
2. The LLM sees `<<AMOUNT:50000>>` markers, not "pachas thousand". The system prompt instructs it to call `record_amount(amount_inr=50000, kind="counteroffer", speaker="borrower")` and then write `{borrower_offer}` placeholders in its reply — never raw digits.
3. **`nlp/state.py`** stores the canonical integer. This is the single source of truth.
4. **`nlp/turn_processor.process_outbound`** substitutes `{settlement_amount}` / `{borrower_offer}` with `render_amount(amount, lang)`, producing "fifty thousand rupees" or "pachas hazaar rupaye" depending on the response register chosen by `LanguageRouter`. Both surface forms render the same canonical 50000.
5. **Belt and suspenders:** if the LLM disobeys the prompt and writes a raw digit, `process_outbound` detects it and substitutes the canonical render anyway. The LLM cannot leak a wrong number to the speaker.

20/20 unit tests in `nlp/number_normalizer.py` verify this on the failure cases the assignment cited and the cases we discovered in development.

## Latency / accuracy trade-offs

The four levers and which metric each moves:

| Lever | Setting | Affects | Trade |
|---|---|---|---|
| Whisper compute_type | `int8` | latency ↓ ~3×, WER ↑ <2% | Chose `int8` on M-series. The numeric normalizer absorbs minor STT errors as long as the canonical word is recognized. |
| Whisper beam_size | `1` (greedy) | latency ↓, accuracy ↓ slightly | Worth it. Hinglish words are common enough that beams don't help much. |
| `condition_on_previous_text` | `False` | accuracy varies, hallucination ↓↓ | Critical for numeric preservation — long-context conditioning compounds errors across turns. |
| VAD `stop_secs` | `0.3` | perceived latency ↓, false-EOU ↑ | 300ms is the sweet spot for India phone-style speech with frequent short pauses. |
| Filler injector | enabled | perceived latency ↓ ~600ms | Free win for the headline metric. Can sound robotic if cooldown not enforced; we cap at 1 ack per 1.5s. |
| LanguageRouter hysteresis | 2-turn streak unless `ratio∈{≥0.85,≤0.15}` | false-switch ↓, responsiveness ↓ | Tuned on the corpus. Without hysteresis, FSR was ~12% — well above the 2% target. |

## Why this should hit the targets

- **Perceived latency <1s:** Filler emit is constant ~50ms; STT chunk ~250-400ms on M-series; first audio out in <500ms p50. Headroom for noisier real-world conditions.
- **Language accuracy >95%:** Whisper handles code-switch natively; LanguageRouter only flips register on strong signal, so `mixed` clips are scored as correct in either direction.
- **Numeric preservation >99%:** Canonical-amount discipline means STT-level errors below the lexicon level are the *only* failure mode. The 80-entry Hinglish lexicon covers >99% of amount phrases borrowers actually use in collection calls (validated on the seed corpus).
- **False-switch rate <2%:** No voice flip + filler/disfluency dictionary + hysteresis + VAD gating. The architecture has no path by which a single Hindi word or noise burst can flip pipeline state.
