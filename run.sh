#!/usr/bin/env bash
# Starts the full LiveKit voice agent pipeline: LiveKit server + Redis (docker compose), vLLM
# (LLM inference), the Chatterbox Turbo TTS microservice (reused unchanged, as its own
# process/venv, from the original Pipeline), the token server (serves web/ and issues LiveKit
# access tokens), and the agent worker. Stop everything with Ctrl+C.
#
# Machine-specific: the vLLM/Chatterbox paths below assume this box's layout (see
# ../Pipeline/_launch_vllm.sh and ../Pipeline/run.sh, which this mirrors) — adjust if you're
# running this somewhere else.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_PY=".venv/bin/python"

MINICONDA_BIN="/home/nauyan/miniconda3/bin"
VLLM_BIN="$MINICONDA_BIN/vllm"
VLLM_MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"
VLLM_SERVED_NAME="qwen2.5-14b-awq"   # must match VLLM_MODEL in .env
VLLM_PORT=8000
VLLM_LOG="logs/vllm.log"
# Same CUDA env this box needs for vLLM (see ../Pipeline/_launch_vllm.sh) — without it, vLLM's
# engine core fails to start ("libnvrtc.so.13: cannot open shared object file: No such file or
# directory") because the toolkit isn't on the default library path here.
VLLM_CUDA_HOME="/home/nauyan/miniconda3/lib/python3.13/site-packages/nvidia/cu13"
# faster-whisper (ctranslate2) needs libcublas.so.12 on the library path too — this venv has no
# nvidia-cublas-cu12 pip wheel of its own, so it silently OOMs-out at *inference* time (not model
# load time) with "Library libcublas.so.12 is not found or cannot be loaded" unless pointed at a
# CUDA 12 install. Ollama happens to bundle one on this box; any CUDA 12 cublas works.
WHISPER_CUDA_LIB_DIR="/usr/local/lib/ollama/cuda_v12"

OLD_PIPELINE_DIR="../Pipeline"   # Chatterbox stays exactly as its own process/venv there
CHATTERBOX_PY="$OLD_PIPELINE_DIR/.chatterbox-venv/bin/python3"
CHATTERBOX_SCRIPT="$OLD_PIPELINE_DIR/src/chatterbox_server.py"
CHATTERBOX_PORT=8766
CHATTERBOX_LOG="logs/chatterbox.log"

TOKEN_SERVER_PORT="${TOKEN_SERVER_PORT:-3000}"
TOKEN_SERVER_LOG="logs/token_server.log"

mkdir -p logs

pids=()
cleanup() {
    echo
    echo "Stopping services..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    docker compose down
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

echo "Starting vLLM (LLM inference)..."
(
    export CUDA_HOME="$VLLM_CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$MINICONDA_BIN:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
    # Skip flashinfer's JIT-compiled sampler (its bundled CCCL headers conflict with the
    # toolkit headers on this box) -- falls back to vLLM's built-in sampler, no JIT needed.
    export VLLM_USE_FLASHINFER_SAMPLER=0
    exec "$VLLM_BIN" serve "$VLLM_MODEL" \
        --served-model-name "$VLLM_SERVED_NAME" \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --gpu-memory-utilization 0.5 --max-model-len 8192 \
        --port "$VLLM_PORT"
) > "$VLLM_LOG" 2>&1 &
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
"$CHATTERBOX_PY" -u "$CHATTERBOX_SCRIPT" > "$CHATTERBOX_LOG" 2>&1 &
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

echo "Starting token server (also serves web/) on port $TOKEN_SERVER_PORT..."
TOKEN_SERVER_PORT="$TOKEN_SERVER_PORT" "$VENV_PY" -u src/token_server.py > "$TOKEN_SERVER_LOG" 2>&1 &
token_pid=$!
pids+=("$token_pid")

until curl -s -o /dev/null "http://localhost:$TOKEN_SERVER_PORT/"; do
    if ! kill -0 "$token_pid" 2>/dev/null; then
        echo "Token server failed to start — see $TOKEN_SERVER_LOG"
        exit 1
    fi
    sleep 1
done
echo "Token server ready."

echo "Open http://localhost:$TOKEN_SERVER_PORT in your browser (use this box's LAN/Tailscale"
echo "address instead of localhost if you're connecting from another machine)."
echo "Starting the agent worker (STT + LLM + TTS + tools)..."
# Deliberately not `exec`'d: exec would replace this shell (and its EXIT trap) with the worker
# process, so Ctrl+C would kill only the worker and orphan vLLM/Chatterbox/token-server.
LD_LIBRARY_PATH="$WHISPER_CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" "$VENV_PY" -u src/worker.py start
