# Decision Journal — Madhav Kataria

## Day 1 - Architectural Decisions

I decided to move ahead with OpenAI gpt-4o-mini because it's a blend of fast LLM and good response. Later switched to gpt-5.4-mini once it became available — better tool-call discipline on the no-rupee-digits-in-text rule. The major problem came when I wanted to pick the TTS and STT. In the start I was thinking of using Unmute (Kyutai Moshi) TTS and STT but it failed because it's biased towards English — even with tuning it didn't perform well on Hinglish. So I moved ahead with STT - Whisper and TTS - Sarvam AI (since it provides a unique blend of English and Hindi and sounds more natural).

Apart from pipeline creation I faced some bugs around the `OutboundTurnProcessor` — first version buffered LLM tokens and then emitted a plain `TextFrame` which caused the assistant context aggregator to see doubled output. The bot's history then had `{settlement_amount}` placeholders in it, which the LLM started echoing the next turn. Took me three iterations: buffered TextFrame (broken) → TTS rewriter only (audio fine but transcript still leaked) → in-place mutation of `LLMTextFrame.text` (works).

Local Whisper was 9-10s/turn.
Tried large-v3 int8 on my laptop — too slow.
Same with large-v3-turbo — better latency but worse Hindi numeric quality on spot checks.
Then deployed `large-v3` (full model, float16) on the A40 server over an SSH tunnel and the latency dropped to ~370ms warm.

Another problem I faced was that Whisper was detecting Urdu on Hinglish input and emitting Arabic-script garbage that corrupted names through the LLM. Pinned `WHISPER_LANGUAGE=hi` to fix it. That created a second-order problem though — in Hindi mode Whisper transliterates English speech into Devanagari ("आई कैन पे" for "I can pay"), so the language router thought every English utterance was Hindi. By curating the `EN_DEVANAGARI_HINTS` lookup (~80 common English words written in Devanagari) the router started classifying these as English and the false-flip rate dropped.

I ran the eval harness and pulled per-clip results to compare directly against Deepgram nova-3. Same 10 amount clips, both pipelines: ours got 8/10 right, Deepgram got 4/10. Wins were on `pachas thousand` (Deepgram heard `purchase thousand`), `chalis hazaar` (Deepgram heard `Cesar final`), and `50k` (Deepgram heard `$5.00 k`). The two we miss are normalizer-lexicon gaps (`peti` for 35, `लाग` for lakh — homophone Whisper transcription error), not architectural — extending the lexicon would close them.

| Audio | Gold | Ours STT → extracted | Baseline STT → extracted |
|---|---|---|---|
| `clip_017` | "I can pay pachas thousand rupees" → [50000] | `iCAN पे 50,000 रुपे,` → **[50000]** ✅ | `I can pay purchase thousand rupee.` → [1000] ❌ |
| `clip_018` | "main pachas thousand de sakta hoon" → [50000] | `मैं 50,000 दे सकता हूँ.` → **[50000]** ✅ | `Ми по час таузин дисактагу.` → [] ❌ (Cyrillic!) |
| `clip_019` | "settle for thirty five thousand please" → [35000] | `settle for 35,000 please.` → **[35000]** ✅ | `Settle for 35,000, please.` → [35000] ✅ |
| `clip_020` | "paintees hazaar mein settle kar lo" → [35000] | `पेटी सजार में सेटल कर लो,` → [] ❌ | `हज़ार में settle कर लो.` → [1000] ❌ |
| `clip_021` | "twenty five thousand abhi aur baaki next month" → [25000] | `25,000 अभी और बाकी next month,` → **[25000]** ✅ | `25,000 next month.` → [25000] ✅ |
| `clip_022` | "ek lakh bahut zyada hai forty thousand chalega" → [100000, 40000] | `एक लाग बहुत ज्यादा है 40,000 चलेगा.` → [1, 40000] ❌ | `एक लाख बहुत ज़्यादा है, 40000 चलेगा.` → [100000, 40000] ✅ |
| `clip_023` | "I will pay 50k today" → [50000] | `I will pay 50K today.` → **[50000]** ✅ | `Will pay $5.00 k today.` → [5000] ❌ |
| `clip_024` | "chalis hazaar final hai sir" → [40000] | `40,000 final है सर.` → **[40000]** ✅ | `Cesar final` → [] ❌ |
| `clip_025` | "thirty thousand please that's all I have" → [30000] | `30,000 please that's all I have.` → **[30000]** ✅ | `30,000, please. That's all I have.` → [30000] ✅ |
| `clip_026` | "tees hazaar de dunga abhi" → [30000] | `30,000 दे डूंगा अभी,` → **[30000]** ✅ | `हज़ार दे दूंगा अभी.` → [1000] ❌ |

## Day 2 - Completion

The errors now are more about the optimization part. One of the parts: macOS Docker cannot pass the host mic into the container, so it took me around 30 min to check the documentation and figure out what was happening. Pivoted to the browser entry point (`agent_web.py`) — FastAPI + SmallWebRTC, browser handles AEC natively, sidesteps the Docker-mic problem entirely. After that it was time to enable unit tests and figure out what was working best, so I generated many sample examples (the corpus in `eval/test_corpus/`) and using those I pinpointed the issues and fixed other things to make it a more reliable bot. It performed materially better than the baseline given to me — 4.5× lower perceived latency (552 ms p50 vs 2495 ms) and 2× the numeric preservation rate.

The last big bug I hit was the doubled-assistant-turn pattern in the prebuilt UI — every new bot reply showed the previous turn's last sentence prefixed onto it. Spent a while thinking it was the LLM, then thought it was my OutboundTurnProcessor, finally read upstream Pipecat's `RTVIObserver` and found `_bot_transcription` is never reset on `LLMFullResponseStartFrame` — there's even a TODO comment from the maintainers about deprecating that path. Patched it locally at module load in `agent.py`.

## One thing I intentionally chose NOT to build

ASR fine-tune on a custom collection-call corpus. Collection-call utterances are a narrow distribution and 200-500 labelled samples would give a meaningful WER lift on the failure cases I'm currently missing (`peti`, `लाग`, etc.). But it's a 2-3 day side-quest with data labelling overhead, and the canonical-amount discipline already absorbs most STT-level errors via the lexicon — so the marginal value of better STT *on amounts specifically* is small. Out of scope for this submission's timeline. If I were taking this to production this is the next thing I'd build.
