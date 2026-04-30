"""
Hinglish debt-collection voice agent — Pipecat 1.1 pipeline entrypoint.

Single-command run:
    make run        # uv-managed venv
    python agent.py # if deps already installed

Wire-up:

    LocalAudioTransport.input  (mic + Silero VAD)
        -> WhisperSTTService                       (OSS, faster-whisper)
        -> InboundTurnProcessor                    (number normalize + lang router)
        -> FillerProcessor                         (emits backchannel at EOU)
        -> LLMContextAggregatorPair.user
        -> OpenAILLMService(gpt-4o-mini)           (streaming, tool: record_amount)
        -> OutboundTurnProcessor                   (placeholder substitution)
        -> SarvamTTSService                        (Bulbul-v2, streaming)
        -> LocalAudioTransport.output  (speaker)
        -> LLMContextAggregatorPair.assistant
"""
from __future__ import annotations

import asyncio
import os
import time
import warnings
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Pipecat 1.1's interruption broadcast occasionally schedules a
# FrameProcessor.__process_frame_task_handler coroutine that gets cancelled
# before it's awaited. Harmless — the surrounding task is cancelled — but
# Python's GC prints a RuntimeWarning. Silence just that one.
warnings.filterwarnings(
    "ignore",
    message=r"coroutine 'FrameProcessor\.__process_frame_task_handler' was never awaited",
    category=RuntimeWarning,
)


# --------------------------------------------------------------------------- #
# Upstream patch: RTVIObserver bot-transcription leak across turns
# --------------------------------------------------------------------------- #
# Pipecat's RTVIObserver._handle_llm_text_frame accumulates LLMTextFrame text
# in self._bot_transcription and only resets it when match_endofsentence
# fires inside the buffer (see processors/frameworks/rtvi/observer.py:530-544
# — note the TODO from the maintainers about deprecating this path).
#
# It does NOT reset on LLMFullResponseStartFrame / LLMFullResponseEndFrame.
# So if a turn's last LLM chunk leaves any residual fragment in the buffer
# (e.g. trailing whitespace from our placeholder flush, or an incomplete
# sentence the LLM ended on), that residual gets *prepended* to the next
# turn's transcription. The visible symptom in the prebuilt UI is that
# every assistant message starts with the last sentence of the previous one.
#
# We monkey-patch on_push_frame to reset the buffer at LLM-response start.
# The reset happens *before* the original implementation runs, so the
# BotLLMStartedMessage RTVI event still fires normally.
def _patch_rtvi_observer_reset_on_llm_start() -> None:
    from pipecat.frames.frames import LLMFullResponseStartFrame
    from pipecat.processors.frameworks.rtvi.observer import RTVIObserver

    if getattr(RTVIObserver, "_madhav_reset_patch_applied", False):
        return

    _orig = RTVIObserver.on_push_frame

    async def _patched(self: Any, data: Any) -> Any:
        if isinstance(data.frame, LLMFullResponseStartFrame):
            # Drop any residual text from the previous response; without
            # this the next turn's transcription is prefixed by the leak.
            self._bot_transcription = ""
        return await _orig(self, data)

    RTVIObserver.on_push_frame = _patched
    RTVIObserver._madhav_reset_patch_applied = True


_patch_rtvi_observer_reset_on_llm_start()


# Defer pipecat imports so unit tests of nlp/* work without pipecat installed.
def _import_pipecat() -> dict[str, Any]:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        Frame,
        InputAudioRawFrame,
        LLMRunFrame,
        StartFrame,
        TextFrame,
        TranscriptionFrame,
        TTSSpeakFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.turns.user_start.vad_user_turn_start_strategy import (
        VADUserTurnStartStrategy,
    )
    from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
        SpeechTimeoutUserTurnStopStrategy,
    )
    from pipecat.turns.user_turn_processor import UserTurnProcessor
    from pipecat.turns.user_turn_strategies import (
        ExternalUserTurnStrategies,
        UserTurnStrategies,
    )
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    return locals()


# --------------------------------------------------------------------------- #
# Frame processors
# --------------------------------------------------------------------------- #

def build_bot_audio_gate(grace_s: float = 0.4):
    """Drop mic audio while the bot is speaking (and for ``grace_s`` after).

    Why: without proper acoustic echo cancellation, the bot's TTS leaks into
    the mic and re-triggers VAD, causing repeated user-turn-start events
    while the bot is mid-sentence. That spawns a storm of interruption
    broadcasts, which (a) makes the bot keep stopping itself and (b) makes
    Whisper hallucinate Hindi credit-rolls on the leaked TTS audio.

    Conceptually this is the same job as Kyutai Unmute's audio frontend:
    bot's own audio must never reach speech recognition. We don't run AEC,
    we just gate.

    The gate sits BEFORE VADProcessor in the pipeline so VAD never sees
    bot-leakage audio at all — no false user-started events, no
    interruption thrash.
    """
    pc = _import_pipecat()
    FrameProcessor = pc["FrameProcessor"]
    BotStartedSpeakingFrame = pc["BotStartedSpeakingFrame"]
    BotStoppedSpeakingFrame = pc["BotStoppedSpeakingFrame"]
    InputAudioRawFrame = pc["InputAudioRawFrame"]

    class BotAudioGate(FrameProcessor):
        def __init__(self) -> None:
            super().__init__()
            self._bot_speaking = False
            self._bot_stopped_ts: float = 0.0

        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_speaking = False
                self._bot_stopped_ts = time.time()

            # Drop audio frames while bot is speaking or within grace window.
            # Every other frame type (control, system, transcription) passes
            # through unchanged.
            if isinstance(frame, InputAudioRawFrame):
                if self._bot_speaking or (
                    time.time() - self._bot_stopped_ts < grace_s
                ):
                    return
            await self.push_frame(frame, direction)

    return BotAudioGate()


def build_inbound_processor(state, router, latency_log):
    pc = _import_pipecat()
    FrameProcessor = pc["FrameProcessor"]
    TranscriptionFrame = pc["TranscriptionFrame"]
    UserStoppedSpeakingFrame = pc["UserStoppedSpeakingFrame"]
    from nlp.turn_processor import TurnLatency, process_inbound

    class InboundTurnProcessor(FrameProcessor):
        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, UserStoppedSpeakingFrame):
                latency_log.append(TurnLatency.now())
            if isinstance(frame, TranscriptionFrame) and frame.text:
                result = process_inbound(frame.text, state, router)
                # Replace transcript with normalized text so the LLM sees
                # <<AMOUNT:N>> tags. Also prepend an authoritative
                # [REPLY_LANG:hi|en] directive computed by the router from
                # word counts. The system prompt instructs the LLM to honor
                # this tag — that's how we get deterministic language
                # switching ("5+ words mostly Hindi -> reply in Hindi";
                # "1-2 Hindi words in an English sentence -> stay English").
                reply_lang = router.reply_lang()
                frame.text = (
                    f"[REPLY_LANG:{reply_lang}] {result.normalized_text}"
                )
            await self.push_frame(frame, direction)

    return InboundTurnProcessor()


def build_outbound_processor(state):
    """In-pipeline placeholder substitution on LLMTextFrames.

    Why this exists, given that SarvamTTSService already runs a
    ``text_rewriter`` on the audio path:

    The TTS rewriter only fixes what's *spoken*. The same LLMTextFrame
    text continues downstream to (a) the assistant context aggregator,
    which records the literal ``{settlement_amount}`` into chat history
    so the LLM sees its own placeholder on the next turn and may keep
    emitting them, and (b) any UI transcript surface. We need substitution
    to happen *in place on the LLMTextFrame* so every downstream consumer
    sees the canonical amount.

    Implementation note: we **mutate** ``frame.text`` rather than emit a
    new frame, and we use ``LLMTextFrame`` (not plain ``TextFrame``) so
    Pipecat's TTS sentence aggregator and the context aggregator both
    pick it up as the single source of truth — no dual emission, no
    type-mismatch hole that would let the original literal escape.
    """
    pc = _import_pipecat()
    FrameProcessor = pc["FrameProcessor"]
    FrameDirection = pc["FrameDirection"]
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
    )
    from nlp.turn_processor import process_outbound

    def _safe_emit_position(buf: str) -> int:
        """Largest prefix length we can release without splitting a placeholder.

        Streaming tokens may chop ``{settlement_amount}`` across frames.
        Hold from the last unmatched ``{`` onward.
        """
        last_open = buf.rfind("{")
        if last_open == -1:
            return len(buf)
        if buf.rfind("}") > last_open:
            return len(buf)
        return last_open

    class OutboundTurnProcessor(FrameProcessor):
        def __init__(self) -> None:
            super().__init__()
            self._buf = ""

        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, LLMFullResponseStartFrame):
                self._buf = ""
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, LLMFullResponseEndFrame):
                # Flush remainder (even if it contains an unclosed '{').
                if self._buf:
                    substituted = process_outbound(self._buf, state)
                    self._buf = ""
                    await self.push_frame(
                        LLMTextFrame(substituted), direction
                    )
                await self.push_frame(frame, direction)
                return

            if (
                isinstance(frame, LLMTextFrame)
                and direction == FrameDirection.DOWNSTREAM
                and frame.text
            ):
                self._buf += frame.text
                cut = _safe_emit_position(self._buf)
                if cut <= 0:
                    # Whole buffer is inside an open placeholder — hold.
                    return
                ready, self._buf = self._buf[:cut], self._buf[cut:]
                substituted = process_outbound(ready, state)
                # Mutate the in-flight frame instead of creating a new one,
                # so there is exactly one LLMTextFrame per chunk on the wire.
                frame.text = substituted
                await self.push_frame(frame, direction)
                return

            await self.push_frame(frame, direction)

    return OutboundTurnProcessor()


def build_filler_processor(state, filler_injector):
    pc = _import_pipecat()
    FrameDirection = pc["FrameDirection"]
    FrameProcessor = pc["FrameProcessor"]
    TranscriptionFrame = pc["TranscriptionFrame"]
    UserStartedSpeakingFrame = pc["UserStartedSpeakingFrame"]
    UserStoppedSpeakingFrame = pc["UserStoppedSpeakingFrame"]
    TTSSpeakFrame = pc["TTSSpeakFrame"]

    class FillerProcessor(FrameProcessor):
        """Emit a TTSSpeakFrame with a short backchannel right at EOU.

        Only fires after a real transcribed turn — never on silence/noise
        that VAD false-tripped on. We watch for a TranscriptionFrame between
        UserStarted and UserStopped; if none arrived, we suppress the filler.
        """

        def __init__(self) -> None:
            super().__init__()
            self._user_started_ts: float | None = None
            self._last_user_text: str = ""
            self._got_transcript_this_turn: bool = False

        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, UserStartedSpeakingFrame):
                self._user_started_ts = time.time()
                self._got_transcript_this_turn = False
            elif isinstance(frame, TranscriptionFrame) and frame.text:
                self._last_user_text = frame.text
                self._got_transcript_this_turn = True
            elif isinstance(frame, UserStoppedSpeakingFrame):
                if self._got_transcript_this_turn:
                    duration = (
                        time.time() - self._user_started_ts
                        if self._user_started_ts
                        else 0.0
                    )
                    lang = state.current_lang
                    pick = filler_injector.maybe_pick(
                        lang=lang if lang in ("hi", "en", "mixed") else "en",
                        user_turn_duration_s=duration,
                        now_ts=time.time(),
                        user_text=self._last_user_text,
                    )
                    if pick:
                        logger.debug(f"Filler [{lang}]: {pick!r}")
                        await self.push_frame(
                            TTSSpeakFrame(text=pick), FrameDirection.DOWNSTREAM
                        )
            await self.push_frame(frame, direction)

    return FillerProcessor()


# --------------------------------------------------------------------------- #
# LLM tool-call handling
# --------------------------------------------------------------------------- #

def register_tools(llm, state):
    """Register tool handlers with the OpenAI LLM service.

    Pipecat 1.1: each handler takes a single FunctionCallParams arg.
    """
    from nlp.state import AmountEvent

    async def record_amount(params) -> None:
        try:
            arguments = params.arguments
            amount = int(arguments["amount_inr"])
            kind = arguments.get("kind", "counteroffer")
            speaker = arguments.get("speaker", "agent")
            state.record(
                AmountEvent(
                    amount_inr=amount,
                    kind=kind,
                    speaker=speaker,
                    turn_idx=state.turn_idx,
                )
            )
            logger.info(f"Tool record_amount: ₹{amount} ({kind}, by {speaker})")
            await params.result_callback(
                {"recorded": True, "amount_inr": amount, "kind": kind}
            )
        except Exception as e:
            logger.exception("record_amount tool error")
            await params.result_callback({"error": str(e)})

    async def update_borrower_name(params) -> None:
        try:
            new_name = str(params.arguments["name"]).strip()
            if not new_name:
                await params.result_callback(
                    {"error": "name was empty"}
                )
                return
            old_name = state.borrower_name
            state.borrower_name = new_name
            state.name_confirmed = True
            logger.info(f"Tool update_borrower_name: {old_name!r} -> {new_name!r}")
            await params.result_callback(
                {"updated": True, "borrower_name": new_name}
            )
        except Exception as e:
            logger.exception("update_borrower_name tool error")
            await params.result_callback({"error": str(e)})

    llm.register_function("record_amount", record_amount)
    llm.register_function("update_borrower_name", update_borrower_name)


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #

def build_task(transport, *, use_bot_audio_gate: bool = True):
    """Construct a fully wired PipelineTask around the given transport.

    Shared between entry points: ``agent.py`` (local mic via
    LocalAudioTransport, gate ON) and ``agent_web.py`` (browser via
    SmallWebRTCTransport, gate OFF because the browser does AEC).

    Returns ``(task, latency_log)``. Caller runs it with PipelineRunner
    and is responsible for dumping the latency log on shutdown.
    """
    pc = _import_pipecat()
    SileroVADAnalyzer = pc["SileroVADAnalyzer"]
    VADParams = pc["VADParams"]
    VADProcessor = pc["VADProcessor"]
    UserTurnProcessor = pc["UserTurnProcessor"]
    UserTurnStrategies = pc["UserTurnStrategies"]
    ExternalUserTurnStrategies = pc["ExternalUserTurnStrategies"]
    VADUserTurnStartStrategy = pc["VADUserTurnStartStrategy"]
    SpeechTimeoutUserTurnStopStrategy = pc["SpeechTimeoutUserTurnStopStrategy"]
    OpenAILLMService = pc["OpenAILLMService"]
    LLMContext = pc["LLMContext"]
    LLMContextAggregatorPair = pc["LLMContextAggregatorPair"]
    LLMUserAggregatorParams = pc["LLMUserAggregatorParams"]
    LLMRunFrame = pc["LLMRunFrame"]
    Pipeline = pc["Pipeline"]
    PipelineTask = pc["PipelineTask"]
    PipelineParams = pc["PipelineParams"]

    from nlp.filler import FillerInjector
    from nlp.language_router import LanguageRouter
    from nlp.state import ConversationState
    from nlp.system_prompt import build_system_prompt, build_tools_schema
    from stt.whisper_stt import WhisperSTTService

    # State. Borrower name comes from CRM in production; for local demo,
    # set BORROWER_NAME=<your name> in .env so the greeting is correct.
    state = ConversationState(
        borrower_name=os.getenv("BORROWER_NAME", "Rajesh"),
        principal_inr=int(os.getenv("PRINCIPAL_INR", "50000")),
    )
    router = LanguageRouter(initial="en")
    # Filler/backchannel defaults to OFF: in practice the random "haan ji"
    # interjections feel like the bot interrupting the user mid-thought.
    # Set FILLER_ENABLED=true to re-enable when measuring perceived latency.
    filler = FillerInjector(
        enabled=os.getenv("FILLER_ENABLED", "false").lower() == "true"
    )
    latency_log: list[Any] = []

    # 500ms VAD silence + 600ms post-VAD speech-timeout means the user has
    # ~1.1s to take a breath mid-thought before the bot decides they're done.
    # With 300ms (the previous setting) the bot kept cutting in on natural
    # pauses, which is the "too much interruption" complaint.
    vad_silence_ms = int(os.getenv("VAD_SILENCE_MS", "500"))
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(stop_secs=vad_silence_ms / 1000.0)
        )
    )

    # User turn lifecycle: VAD-based start, speech-timeout stop. We avoid the
    # default smart-turn analyzer because it needs an extra ONNX model and
    # isn't necessary for our latency targets — VAD silence is the trigger.
    #
    # user_turn_stop_timeout is the hard ceiling that fires if no transcript
    # arrives. Override via USER_TURN_STOP_TIMEOUT_S; we keep this small (0.5s)
    # so a no-transcript turn doesn't wedge the pipeline for seconds. STT
    # itself is the long pole; trim the ceiling on top of it.
    user_speech_timeout = float(os.getenv("USER_SPEECH_TIMEOUT_S", "0.6"))
    user_turn_stop_timeout = float(os.getenv("USER_TURN_STOP_TIMEOUT_S", "0.5"))
    user_turn = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(
                user_speech_timeout=user_speech_timeout,
            )],
        ),
        user_turn_stop_timeout=user_turn_stop_timeout,
    )

    # STT (OSS) — faster-whisper large-v3 int8
    stt = WhisperSTTService()

    # LLM (closed). gpt-4o is the default — materially better than gpt-4o-mini
    # at instruction-following on the no-rupee-digits-in-text rule and at
    # mirroring the user's Hinglish register. Set OPENAI_MODEL=gpt-5.1 (or any
    # other Chat Completions-compatible OpenAI model id) to override.
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
    logger.info(f"Using OpenAI model: {openai_model}")
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(model=openai_model),
    )
    register_tools(llm, state)

    # TTS — Sarvam preferred, Cartesia fallback
    tts = _build_tts(state)

    # Conversation context. We pass ExternalUserTurnStrategies so the
    # aggregator does NOT spin up its own duplicate VAD/transcription turn
    # controller — it consumes the UserStarted/StoppedSpeakingFrame our
    # explicit UserTurnProcessor already emits. Without this, both run in
    # parallel and race on broadcast_interruption (the dangling-coroutine
    # RuntimeWarning we saw).
    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(state)}],
        tools=build_tools_schema(),
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=ExternalUserTurnStrategies(),
        ),
    )

    # Bot-audio gate is needed when there's no browser-side AEC (LocalAudio
    # path). For SmallWebRTCTransport, the browser cancels echo natively, and
    # stacking the gate on top discards real user speech that arrives during
    # TTS — so the web entry point passes use_bot_audio_gate=False.
    processors = [transport.input()]
    if use_bot_audio_gate:
        processors.append(build_bot_audio_gate(grace_s=0.4))
    processors.extend([
        vad,                  # emits VADUserStarted/StoppedSpeakingFrame
        user_turn,            # emits UserStarted/StoppedSpeakingFrame
        stt,                  # SegmentedSTTService — one Whisper call per turn
        build_inbound_processor(state, router, latency_log),
        build_filler_processor(state, filler),
        context_aggregator.user(),
        llm,
        # Mutate LLMTextFrames in place so {settlement_amount}/{borrower_offer}
        # never reach the assistant context aggregator (which would store the
        # literal in chat history and the LLM would echo it next turn) or any
        # UI transcript. The TTS-side text_rewriter is defense-in-depth on the
        # audio path; this is the source-of-truth fix for everything else.
        build_outbound_processor(state),
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
    pipeline = Pipeline(processors)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=22_050,
        ),
    )

    @task.event_handler("on_pipeline_started")
    async def _on_started(_task, _frame):  # noqa: ANN001
        # Kick off the call by asking the LLM to produce its greeting. We
        # don't hand-write the greeting any more — the system prompt already
        # tells Priya how to open, and letting the LLM produce it means
        # English names like "Rajesh" go through en-IN TTS and get
        # pronounced correctly.
        await task.queue_frame(LLMRunFrame())

    return task, latency_log


async def run() -> None:
    """Local-mic entry point. Browser entry point lives in agent_web.py."""
    pc = _import_pipecat()
    LocalAudioTransport = pc["LocalAudioTransport"]
    LocalAudioTransportParams = pc["LocalAudioTransportParams"]
    PipelineRunner = pc["PipelineRunner"]

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=22_050,
        )
    )
    task, latency_log = build_task(transport, use_bot_audio_gate=True)
    runner = PipelineRunner()
    logger.info("Agent ready. Speak into the mic. Ctrl+C to stop.")
    try:
        await runner.run(task)
    finally:
        _dump_latency(latency_log)


def _build_tts(state):
    from nlp.turn_processor import process_outbound

    # Closure captures `state`. Sarvam calls this on each sentence-level
    # chunk right before synthesis. This is the authoritative placeholder
    # → canonical-amount substitution; the OutboundTurnProcessor does the
    # same job earlier in the pipeline as defense-in-depth.
    def _rewrite(text: str) -> str:
        return process_outbound(text, state)

    if os.getenv("SARVAM_API_KEY"):
        from tts.sarvam_tts import SarvamTTSService

        return SarvamTTSService(
            target_language_code="hi-IN",
            text_rewriter=_rewrite,
        )
    if os.getenv("CARTESIA_API_KEY"):
        from tts.cartesia_fallback import make_cartesia_tts

        logger.warning("Falling back to Cartesia TTS (less natural Hinglish)")
        return make_cartesia_tts()
    raise RuntimeError(
        "No TTS available. Set SARVAM_API_KEY (preferred) or CARTESIA_API_KEY."
    )


def _dump_latency(latency_log: list[Any]) -> None:
    if not latency_log:
        return
    perceived = [
        t.perceived_latency_ms()
        for t in latency_log
        if t.perceived_latency_ms() is not None
    ]
    if not perceived:
        return
    perceived.sort()
    p50 = perceived[len(perceived) // 2]
    p95 = perceived[int(len(perceived) * 0.95)] if len(perceived) > 1 else p50
    logger.info(
        f"Perceived latency over {len(perceived)} turns: p50={p50:.0f}ms p95={p95:.0f}ms"
    )


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Bye.")
