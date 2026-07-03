#!/usr/bin/env bash
# Starts the full voice agent pipeline: Chatterbox TTS microservice, the main
# STT+LLM+WebSocket server, and a static file server for index.html.
# Stop everything with Ctrl+C.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MINICONDA_PY="/home/nauyan/miniconda3/bin/python3"
CHATTERBOX_PY=".chatterbox-venv/bin/python3"
CHATTERBOX_LOG="chatterbox.log"
HTTP_PORT=3000

pids=()
cleanup() {
    echo
    echo "Stopping services..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

if ! curl -s -o /dev/null http://localhost:11434/api/tags; then
    echo "Warning: Ollama doesn't seem to be reachable on localhost:11434 — start it with 'ollama serve'."
fi

echo "Starting Chatterbox Turbo TTS service..."
"$CHATTERBOX_PY" -u chatterbox_server.py > "$CHATTERBOX_LOG" 2>&1 &
pids+=($!)

echo "Waiting for Chatterbox to be ready (this can take ~10-60s)..."
until grep -q "Chatterbox Turbo ready" "$CHATTERBOX_LOG" 2>/dev/null; do
    if ! kill -0 "${pids[0]}" 2>/dev/null; then
        echo "Chatterbox failed to start — see $CHATTERBOX_LOG"
        exit 1
    fi
    sleep 1
done
echo "Chatterbox ready."

echo "Starting static file server on port $HTTP_PORT..."
python3 -m http.server "$HTTP_PORT" > /dev/null 2>&1 &
pids+=($!)

echo "Open http://localhost:$HTTP_PORT in your browser (use 127.0.0.1 if 'localhost' misbehaves)."
echo "Starting main server (STT + LLM + WebSocket)..."
"$MINICONDA_PY" -u server.py
