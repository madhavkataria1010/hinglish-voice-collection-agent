.PHONY: install run web baseline eval demo clean docker-build docker-run tunnel

install:
	uv sync --all-extras

run:
	uv run python agent.py

# Browser frontend. Opens FastAPI on http://localhost:7860/ — go there in
# Chrome, click Connect, and talk into your laptop mic. Uses WebRTC, so
# the browser handles AEC; STT still hits the A40 over the SSH tunnel.
web:
	uv run python agent_web.py

# Open / refresh the SSH tunnel to the A40 STT server. If the remote
# uvicorn is dead, restart it before opening the tunnel. Idempotent.
tunnel:
	@echo "[1/3] Checking remote uvicorn..."; \
	if ! ssh a40 "pgrep -f 'uvicorn server:app' > /dev/null"; then \
	  echo "  remote uvicorn down — restarting (model load ~75s)"; \
	  ssh a40 "cd /home/kartik/madhav/Industry/whisper-server && setsid nohup ./start.sh </dev/null >server.log 2>&1 & sleep 1" >/dev/null; \
	  echo -n "  waiting for /health"; \
	  for i in $$(seq 1 60); do \
	    if ssh a40 "curl -s --max-time 2 http://localhost:8765/health" 2>/dev/null | grep -q '"ok":true'; then echo " ready"; break; fi; \
	    echo -n "."; sleep 2; \
	  done; \
	else echo "  remote uvicorn is alive"; fi; \
	echo "[2/3] Opening local tunnel..."; \
	pkill -f 'ssh.*-L.*8765:localhost:8765' 2>/dev/null; sleep 1; \
	ssh -fN -L 8765:localhost:8765 a40 && sleep 1; \
	echo "[3/3] Verifying via local tunnel..."; \
	curl -fsS --max-time 5 http://localhost:8765/health && echo "" || \
	  (echo "FAIL — tunnel up but server unreachable. Check: ssh a40 'tail /home/kartik/madhav/Industry/whisper-server/server.log'"; exit 1)

baseline:
	PIPELINE=baseline uv run python agent.py

eval:
	uv run python -m eval.run_eval

eval-baseline:
	uv run python -m eval.run_eval --pipeline baseline

eval-ours:
	uv run python -m eval.run_eval --pipeline ours

demo:
	uv run python -m eval.record_demo

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-run:
	docker compose -f docker/docker-compose.yml up

clean:
	rm -rf .venv __pycache__ */__pycache__ */*/__pycache__ eval/report.md eval/results/

test-normalizer:
	uv run python -m nlp.number_normalizer
