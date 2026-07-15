#!/usr/bin/env bash
# Starts the full voice agent pipeline: Chatterbox TTS microservice, the main
# STT+LLM+WebSocket server, and a static file server for index.html.
# Stop everything with Ctrl+C.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MINICONDA_PY="/home/nauyan/miniconda3/bin/python3"
VLLM_BIN="/home/nauyan/miniconda3/bin/vllm"
VLLM_MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"
VLLM_SERVED_NAME="qwen2.5-14b-awq"   # must match VLLM_MODEL in src/server.py
VLLM_PORT=8000
VLLM_LOG="logs/vllm.log"
# gpu_memory_utilization=0.5 and max_model_len=8192 keep vLLM's footprint bounded so it shares
# the one GPU safely with Whisper (in src/server.py) and Chatterbox below — same conservative
# budgeting precedent as gen_reference.py's BudgetedOrpheusModel for this box.
CHATTERBOX_PY=".chatterbox-venv/bin/python3"
CHATTERBOX_LOG="logs/chatterbox.log"
WEB_DIR="web"
HTTP_PORT=3000

mkdir -p logs

pids=()
cleanup() {
    echo
    echo "Stopping services..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

if ! curl -s -o /dev/null "http://localhost:1234/search?q=test&format=json"; then
    echo "Warning: SearXNG doesn't seem to be reachable on localhost:1234 — web search will fail until it's up."
fi

# Ensure the RAG document index exists before serving. It's a regenerable, gitignored artifact
# (data/rag_index.npy) — but without it, always-on retrieval finds nothing and the agent answers
# "I don't have that on file" for every product/fee/rate/policy question, which looks like the whole
# knowledge base is broken. Build it if missing; refuse to start (loudly) if the build fails.
RAG_INDEX="data/rag_index.npy"
if [ ! -f "$RAG_INDEX" ]; then
    echo "RAG index ($RAG_INDEX) missing — building it from config/rag_docs/ (one-time, ~20s)..."
    if ! "$MINICONDA_PY" -u src/build_index.py; then
        echo "ERROR: failed to build the RAG index — refusing to start. Fix the error above and" >&2
        echo "retry; the agent would otherwise report no information for every business question." >&2
        exit 1
    fi
fi

echo "Starting vLLM (LLM inference, replaces Ollama)..."
"$VLLM_BIN" serve "$VLLM_MODEL" \
    --served-model-name "$VLLM_SERVED_NAME" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --gpu-memory-utilization 0.5 --max-model-len 8192 \
    --port "$VLLM_PORT" > "$VLLM_LOG" 2>&1 &
vllm_pid=$!
pids+=("$vllm_pid")

echo "Waiting for vLLM to be ready (model load can take 30-90s)..."
until curl -s -o /dev/null "http://localhost:$VLLM_PORT/v1/models"; do
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
        echo "vLLM failed to start — see $VLLM_LOG"
        exit 1
    fi
    sleep 2
done
echo "vLLM ready."

echo "Starting Chatterbox Turbo TTS service..."
"$CHATTERBOX_PY" -u src/chatterbox_server.py > "$CHATTERBOX_LOG" 2>&1 &
chatterbox_pid=$!
pids+=("$chatterbox_pid")

echo "Waiting for Chatterbox to be ready (this can take ~10-60s)..."
until grep -q "Chatterbox Turbo ready" "$CHATTERBOX_LOG" 2>/dev/null; do
    if ! kill -0 "$chatterbox_pid" 2>/dev/null; then
        echo "Chatterbox failed to start — see $CHATTERBOX_LOG"
        exit 1
    fi
    sleep 1
done
echo "Chatterbox ready."

echo "Starting static file server on port $HTTP_PORT..."
python3 -m http.server "$HTTP_PORT" --directory "$WEB_DIR" > /dev/null 2>&1 &
pids+=($!)

echo "Open http://localhost:$HTTP_PORT in your browser (use 127.0.0.1 if 'localhost' misbehaves)."
echo "Starting main server (STT + LLM + WebSocket)..."
"$MINICONDA_PY" -u src/server.py
