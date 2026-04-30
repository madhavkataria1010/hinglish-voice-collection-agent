# Remote Whisper STT

Runs `faster-whisper large-v3` on the A40 (`a40` SSH host) and serves it over
HTTP. The agent on your laptop hits it via an SSH tunnel.

## Why

Local Mac CPU int8 was ~9-10s per turn. A40 float16 warm path is ~350ms.

## One-time setup (already done)

```bash
ssh a40 "curl -LsSf https://astral.sh/uv/install.sh | sh"
scp server.py requirements.txt start.sh a40:/home/kartik/madhav/Industry/whisper-server/
ssh a40 "cd /home/kartik/madhav/Industry/whisper-server && \
  /home/kartik/.local/bin/uv venv --python 3.11 .venv && \
  /home/kartik/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt"
```

## Start / restart the server

```bash
ssh a40 "cd /home/kartik/madhav/Industry/whisper-server && \
  pkill -f 'uvicorn server:app'; \
  setsid nohup ./start.sh </dev/null >server.log 2>&1 &"
```

Logs live at `/home/kartik/madhav/Industry/whisper-server/server.log`.

## Open the SSH tunnel from the laptop

```bash
ssh -fN -L 8765:localhost:8765 a40
```

Verify: `curl -s http://localhost:8765/health` →
`{"ok":true,"model":"large-v3","device":"cuda","compute":"float16"}`

## Switch the agent to use it

In `.env`:

```
WHISPER_BACKEND=remote
WHISPER_REMOTE_URL=http://localhost:8765
```

To go back to local STT, set `WHISPER_BACKEND=local`.
