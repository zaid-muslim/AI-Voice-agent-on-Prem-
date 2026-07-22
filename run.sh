#!/usr/bin/env bash
# Starts the always-on pieces of the LiveKit voice agent pipeline: LiveKit server + Redis
# (docker compose) and the token server (serves web/, issues LiveKit access tokens, and hosts
# the model-selection orchestrator). vLLM, the Chatterbox TTS microservice, and the agent worker
# are no longer started here — src/orchestrator.py launches those on demand, once you've picked
# your LLM/STT/TTS in the browser and confirmed (see config/models_config.json for the available
# choices). Stop everything with Ctrl+C.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_PY=".venv/bin/python"
TOKEN_SERVER_PORT="${TOKEN_SERVER_PORT:-3000}"

mkdir -p logs

cleanup() {
    echo
    echo "Stopping services..."
    # --profile on-demand down covers BOTH the always-on services (redis/livekit, no profile —
    # always active regardless of --profile flags) and the on-demand ones (vllm/whisper/
    # chatterbox/worker). This is the last safety net if orchestrator.shutdown_all() (triggered by
    # token_server.py's own shutdown below) hasn't fully torn those down yet, e.g. Ctrl+C landing
    # mid-launch — a bare `docker compose down` would miss the on-demand profile entirely.
    docker compose --profile on-demand down
}
trap cleanup EXIT INT TERM

if ! curl -s -o /dev/null "http://localhost:1234/search?q=test&format=json"; then
    echo "Warning: SearXNG doesn't seem to be reachable on localhost:1234 — web search will fail until it's up."
fi

# RAG index: regenerable, gitignored artifact — without it, always-on retrieval finds nothing
# and the agent answers "I don't have that on file" for every product/fee/rate/policy question.
RAG_INDEX="data/rag_index.npy"
if [ ! -f "$RAG_INDEX" ]; then
    echo "RAG index ($RAG_INDEX) missing — building it from config/rag_docs/ (one-time, ~20s)..."
    if ! "$VENV_PY" -u src/build_index.py; then
        echo "ERROR: failed to build the RAG index — refusing to start. Fix the error above and" >&2
        echo "retry; the agent would otherwise report no information for every business question." >&2
        exit 1
    fi
fi

echo "Starting LiveKit server + Redis (docker compose)..."
docker compose up -d
echo "Waiting for LiveKit to be ready..."
until curl -s -o /dev/null "http://localhost:7880/"; do
    sleep 1
done
echo "LiveKit ready."

echo "Starting token server (also serves web/) on port $TOKEN_SERVER_PORT..."
echo "Open http://localhost:$TOKEN_SERVER_PORT (use this box's LAN/Tailscale address instead of"
echo "localhost if you're connecting from another machine) to pick your models and load the"
echo "pipeline — nothing else starts until you confirm a selection there."
# Foreground, not backgrounded/exec'd: vLLM/Chatterbox/the agent worker are now launched *by*
# token_server.py itself (src/orchestrator.py) post-confirmation from the browser, so this
# process's own shutdown (Ctrl+C -> SIGINT) must be what triggers its FastAPI shutdown hook
# (orchestrator.shutdown_all()) to tear those down before this trap's `docker compose down` runs.
TOKEN_SERVER_PORT="$TOKEN_SERVER_PORT" "$VENV_PY" -u src/token_server.py
