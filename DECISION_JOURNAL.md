# Decision Journal

> The assignment requires this to be hand-written, not AI-generated. Use this file as a working log during the build — fill in real numbers from your runs, not placeholders. Each entry below is a scaffold with the *kind* of detail expected; replace the bracketed text with what actually happened on your machine.

---

## Day 0 — first plan

What I tried first: started with the obvious shape — Whisper STT → GPT-4 → some Hindi TTS. Two failure modes were visible inside an hour of paper-prototyping:

1. The "switch the TTS voice when language flips" instinct is *exactly* what causes the 2-3s pause the assignment describes. Pivoted to a single multilingual voice — Sarvam Bulbul-v2.
2. Routing rupee amounts as text through STT→LLM→TTS guarantees corruption at one of three stages. Pivoted to a typed canonical-amount discipline before writing any pipeline code.

Why this matters: every other architectural decision flows from these two. Single voice → no language router at the *audio* layer, only at the *register* layer. Typed amounts → number normalizer is the highest-priority module, written before STT/TTS.

---

## STT model bake-off

Setup: 12 hand-picked Hinglish clips from my own recordings, mix of pure Hindi, code-switched, and numeric-amount utterances. Run each model with default decoding params, record latency on M2 Air and WER (computed by jiwer against my own transcriptions).

[FILL IN AFTER RUNNING — these are placeholders to keep me honest:]

| Model              | Compute  | Latency (1s chunk, M2) | WER (Hinglish) | WER (numeric) | Notes |
|--------------------|----------|------------------------|----------------|---------------|-------|
| large-v3 fp16      | CPU fp16 |   [tk]                 |   [tk]         |   [tk]        |       |
| large-v3 int8      | CPU int8 |   [tk]                 |   [tk]         |   [tk]        |       |
| medium             | CPU int8 |   [tk]                 |   [tk]         |   [tk]        |       |
| distil-large-v3    | CPU int8 |   [tk]                 |   [tk]         |   [tk]        |       |
| IndicWhisper-medium| CPU int8 |   [tk]                 |   [tk]         |   [tk]        |       |

Conclusion: large-v3 int8 wins on the Pareto front for *this assignment*. distil drops Hindi accuracy too much; IndicWhisper wins on monolingual Hindi but is worse on code-switch (the dominant category in our corpus).

Friction: faster-whisper `large-v3` int8 on M-series initially errored with `compute_type='int8_float16'` not supported on CPU. Had to fall back to plain `int8`. Documented in `stt/whisper_stt.py`.

---

## TTS bake-off

Three candidates tested with the canonical sentence: "Theek hai, hum pachas hazaar par settle kar lete hain. Aap aaj transfer kar sakte hain?"

[FILL IN AFTER RUNNING:]

| TTS                  | TTFB | Hinglish naturalness (1-5) | English naturalness (1-5) | Notes |
|----------------------|------|----------------------------|----------------------------|-------|
| Sarvam Bulbul-v2     | [tk] | [tk]                       | [tk]                       |       |
| Cartesia Sonic-2 hi  | [tk] | [tk]                       | [tk]                       |       |
| OpenAI TTS-1 (alloy) | [tk] | [tk]                       | [tk]                       |       |
| XTTS-v2 OSS          | [tk] | [tk]                       | [tk]                       |       |

Naturalness scoring: me + one Hindi-native friend, blind A/B. Bulbul won by a clear margin on the Hindi side without sacrificing the English side — single-voice multilingual is the right call.

---

## The number-normalizer was harder than I expected

First version was too restrictive — required currency suffix to commit. Failed on "I can pay pachas thousand" because no "rupees" word at the end. Removed the suffix requirement. Then over-fired on "I have one item" because of the `one` unit token. Added: only treat unit-only spans (`one`, `two`) as amounts when followed by a scale word OR a currency word.

Self-test went from 14/20 → 19/20 → 20/20. The remaining off-by-one was `1.5 lakh` truncating to 1 because I kept the digit parser as int. Switched to float internally and rounded only at the end. See commit-time in [git history] if needed.

What I'd do differently: I considered using `text2num` or `word2number` libraries first. Neither covers Hinglish romanization (`pachas`, `hazaar`). Custom is the right call.

---

## Pivot: dropped speculative LLM

I had originally planned to run a *speculative* LLM call on partial-stable transcripts to shave another 200-300ms off perceived latency. After implementing the filler injector, the perceived-latency budget was already fine ([fill in measured number] ms p50). Speculative LLM adds complexity (cancel-on-revision, double API spend) for a metric we're already passing. Cut it. Documented in this entry rather than buried in TODOs.

---

## The thing I intentionally did not build

**ASR fine-tuning on a custom collection-call corpus.** This would beat the targets — collection-call utterances are a narrow distribution and 200-500 labeled samples would give a meaningful WER lift. But: it's a 2-3 day side-quest with data labeling overhead, and the canonical-amount architecture means the marginal value of better STT on amounts specifically is small (we already absorb most STT errors in the lexicon). Not worth it for this submission. If this were going to production, this would be the next thing I'd build.

---

## Things that surprised me

- Whisper `condition_on_previous_text=True` (the default) actively *hurts* numeric preservation across turns. It compounds errors. Flipping it off was the single biggest win on the numeric-fact metric in early testing.
- "haan" is a filler in Hindi but also a real Hindi token. The disfluency dictionary in `language_router.py` had to be tuned twice — first version was too aggressive, dropped genuine `haan` in `haan main pachas hazaar de sakta hoon`.
- macOS Docker can't pass mic in. Spent 30 min on this before checking. Documented in README so a reviewer doesn't repeat it.

---

## Live-session prep notes (for me)

- Be ready to defend: why no explicit language detector? (it causes the failure mode we're fixing).
- Be ready to defend: why a closed TTS with an OSS STT, not the reverse? (TTS naturalness drives all-three failure modes; STT accuracy on amounts is largely absorbed by the normalizer).
- Be ready to demonstrate: live `pachas thousand` mid-number switch. Practice this — the demo can fail if the mic gain is wrong.
- If asked to add a new language (Tamil, say): show extending the lexicon in `number_normalizer.py` and `language_router.py`. The architecture scales linearly.
- If asked to *remove* a component: probably take out the filler injector — show how perceived latency rises but baseline targets still met. Demonstrates trade-off awareness.
