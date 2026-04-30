"""
Browser-based entry point for the Hinglish debt-collection voice agent.

Run:
    make web
    # then open http://localhost:7860/

Architecture:
    Browser (mic + speaker + native AEC)
        <--- WebRTC (audio + datachannel) --->
    FastAPI on localhost:7860
        ├── GET  /              -> redirect to /client/
        ├── /client/*           -> SmallWebRTCPrebuiltUI (static bundle)
        ├── POST /api/offer     -> WebRTC SDP offer/answer
        └── PATCH /api/offer    -> ICE candidate trickle

For each new browser session, a SmallWebRTCConnection is created and
``agent.build_task`` constructs a fresh Pipeline around it. The pipeline
is identical to the local-mic entry point except it skips ``BotAudioGate``
— the browser already does acoustic echo cancellation, and gating on top
of AEC drops real user speech that arrives while the bot is talking.

STT still goes to the remote Whisper server on the A40 over the SSH
tunnel (``WHISPER_BACKEND=remote`` in .env); this entry point doesn't
touch that path.
"""
from __future__ import annotations

import asyncio
import os
import uuid
import warnings

from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: F401  (warm load)
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

# Same warning silencer agent.py uses.
warnings.filterwarnings(
    "ignore",
    message=r"coroutine 'FrameProcessor\.__process_frame_task_handler' was never awaited",
    category=RuntimeWarning,
)

# Pipecat's request_handler module re-exports these dataclasses via the
# `from pipecat.transports.smallwebrtc.request_handler import ...` path,
# but we also need the request schemas for the FastAPI route signatures.
from pipecat.transports.smallwebrtc.request_handler import (  # noqa: E402
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
)

import agent as agent_mod  # local module — provides build_task

load_dotenv()


# Track tasks so we can drain them on shutdown.
_active_runners: set[asyncio.Task] = set()


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    """Drain any in-flight pipelines on shutdown."""
    yield
    if _active_runners:
        logger.info(f"Draining {len(_active_runners)} pipeline runner(s)...")
        for t in list(_active_runners):
            t.cancel()
        await asyncio.gather(*_active_runners, return_exceptions=True)


app = FastAPI(lifespan=_lifespan)
app.mount("/client", SmallWebRTCPrebuiltUI)


@app.get("/", include_in_schema=False)
async def _root() -> RedirectResponse:
    return RedirectResponse(url="/client/")


from pipecat.transports.smallwebrtc.request_handler import (  # noqa: E402
    ConnectionMode,
)

# SINGLE mode: second browser tab can't squat on a stale connection.
_handler = SmallWebRTCRequestHandler(connection_mode=ConnectionMode.SINGLE)


async def _run_pipeline_for(connection: SmallWebRTCConnection) -> None:
    """Build and run a Pipeline for a single browser session."""
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=22_050,
        ),
    )
    # use_bot_audio_gate=False — see agent.build_task docstring. Browser
    # AEC handles echo; stacking the gate drops legit barge-ins.
    task, latency_log = agent_mod.build_task(transport, use_bot_audio_gate=False)
    runner = PipelineRunner()
    logger.info(f"Pipeline started for pc_id={connection.pc_id}")
    try:
        await runner.run(task)
    finally:
        agent_mod._dump_latency(latency_log)
        logger.info(f"Pipeline finished for pc_id={connection.pc_id}")


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def _on_connection(connection: SmallWebRTCConnection) -> None:
        t = asyncio.create_task(_run_pipeline_for(connection))
        _active_runners.add(t)
        t.add_done_callback(_active_runners.discard)

    return await _handler.handle_web_request(
        request=request,
        webrtc_connection_callback=_on_connection,
    )


@app.patch("/api/offer")
async def ice(request: SmallWebRTCPatchRequest):
    await _handler.handle_patch_request(request)
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Pipecat-Cloud / RTVI compatibility shim
# --------------------------------------------------------------------------- #
# The prebuilt UI talks RTVI, which calls:
#   1. POST /start                              -> get a sessionId (+ iceConfig)
#   2. POST /sessions/{sessionId}/api/offer     -> SDP exchange (proxied to /api/offer)
#   3. PATCH /sessions/{sessionId}/api/offer    -> ICE trickle  (proxied)
# Without these, the UI errors with "Not Found" right after entering the
# `authenticating` state. We mirror Pipecat's runner exactly here.

_active_sessions: dict[str, dict[str, Any]] = {}


@app.post("/start")
async def rtvi_start(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_id = str(uuid.uuid4())
    _active_sessions[session_id] = body.get("body", {})
    result: dict[str, Any] = {"sessionId": session_id}
    if body.get("enableDefaultIceServers"):
        result["iceConfig"] = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    return result


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def rtvi_proxy(
    session_id: str,
    path: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    session = _active_sessions.get(session_id)
    if session is None:
        return Response(content="Invalid or not-yet-ready session_id", status_code=404)

    if path.endswith("api/offer"):
        try:
            data = await request.json()
        except Exception as e:
            logger.error(f"Bad JSON on /sessions/.../api/offer: {e}")
            return Response(content="Invalid WebRTC request", status_code=400)

        if request.method == "POST":
            webrtc_req = SmallWebRTCRequest(
                sdp=data["sdp"],
                type=data["type"],
                pc_id=data.get("pc_id"),
                restart_pc=data.get("restart_pc"),
                request_data=(
                    data.get("request_data")
                    or data.get("requestData")
                    or session
                ),
            )
            return await offer(webrtc_req, background_tasks)
        if request.method == "PATCH":
            patch_req = SmallWebRTCPatchRequest(
                pc_id=data["pc_id"],
                candidates=[IceCandidate(**c) for c in data.get("candidates", [])],
            )
            return await ice(patch_req)

    logger.info(f"RTVI proxy: unhandled {request.method} {path}")
    return Response(status_code=200)


def main() -> None:
    import uvicorn

    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "7860"))
    logger.info(f"Web UI: http://{host}:{port}/")
    uvicorn.run("agent_web:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
