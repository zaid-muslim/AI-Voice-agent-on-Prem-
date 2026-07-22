# Voice Agent Pipeline (LiveKit)

Real-time voice assistant: speech in → transcription → LLM response → cloned-voice speech out,
built on [LiveKit](https://livekit.io/) instead of a bespoke WebSocket/VAD server. Same bank
branch receptionist domain as the original `Pipeline/`: answers informational questions (branches,
hours, services, contact info, and detailed product/fee/rate/policy questions grounded in its own
documents), blocks a lost/stolen card after identity verification, and logs a callback request for
a human. It still cannot check balances or make transactions.

`banking.py`, `db.py`, `rag.py`, `convo_log.py`, and `config/bank_config.json` are reused unchanged
from the original `Pipeline/` — only the transport layer (WebSocket → LiveKit) and the
STT/LLM/TTS wiring changed.

## Architecture

```
Browser  ──WebRTC──▶  LiveKit SFU (self-hosted, Docker)
                            │  job dispatch (WebSocket)
                            ▼
                     Agent worker (src/worker.py)
                     — one process per active call —
                            │  HTTP, all three below
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
       Whisper service   vLLM        Chatterbox service
       (STT, shared)   (LLM, shared)   (TTS, shared)
```

- **LiveKit is the media server**, not this codebase — it handles the actual WebRTC
  signaling/audio transport. `src/worker.py` is a *client* of it: it opens a persistent
  connection, registers itself, and LiveKit dispatches each new call to it as a job.
- **STT, LLM, and TTS are all shared HTTP microservices**, not models loaded inside the call
  process. A call's job process holds no GPU model of its own — it just makes HTTP requests to
  whichever backend is running. This matters for concurrency: earlier, STT was loaded fresh *per
  call*, which OOM'd the GPU on a second simultaneous caller. Making it a shared service (like TTS
  already was) fixed that — see `src/whisper_server.py` / `src/whisper_stt.py`.
- **Nothing loads until you pick it.** `src/token_server.py` serves the browser's model-picker
  screen first; only after you confirm an LLM/STT/TTS choice does `src/orchestrator.py` actually
  start anything, per `config/models_config.json`.
- **The GPU-heavy backends run in Docker** (`docker-compose.yml`, `Dockerfile.whisper`,
  `Dockerfile.worker`, `../Pipeline/Dockerfile.chatterbox`) — `orchestrator.py` launches them as
  `docker compose` services on demand and polls their `/health` endpoints for readiness. This
  isolates their dependencies (CUDA/cuDNN versions, Python packages) from anything else on the
  host, and from other unrelated projects sharing the same GPU box.
- **Scaling out**: a second GPU machine can run its own copy of the agent worker — LiveKit's job
  dispatch already load-balances across every registered worker regardless of which machine it's
  on, no code change needed for that part. Whisper/Chatterbox can similarly run as a *pool* of
  instances across both boxes, picked round-robin per call. See `plans/` for the in-progress
  design.

## Domain config

All brand/company-specific data lives in `config/bank_config.json` (bank name, branches, hours,
services, contact info) — re-read on every turn, no restart needed. The bot greets automatically
the moment a call connects, using the same LLM+TTS pipeline as any other turn.

## Repository layout

```
src/            worker.py (LiveKit agent — STT/LLM/TTS orchestration per call),
                token_server.py (auth + model-picker + static web/ host),
                orchestrator.py (launches/tears down backends via Docker Compose),
                whisper_server.py + whisper_stt.py (shared STT service + its client plugin),
                chatterbox_tts.py (TTS client plugin — server lives in ../Pipeline/src/),
                banking.py, db.py, rag.py, convo_log.py, build_index.py, seed_db.py, show_db.py
web/            Browser frontend (livekit-client) — index.html, vendor/
config/         bank_config.json (swappable domain config), models_config.json (backend
                catalog — id/label/launch params per LLM/STT/TTS option), rag_docs/
data/           bank.db, memory.json, rag_index.npy — gitignored, regenerable/mutable
docker-compose.yml, Dockerfile.whisper, Dockerfile.worker
                Container definitions for the on-demand backends (see Architecture above)
livekit.yaml    Self-hosted LiveKit server config
tests/          pytest suite (banking/RAG logic)
plans/          Design/planning docs
logs/           Runtime logs, plus each backend's Docker container log — gitignored
run.sh          Starts LiveKit+Redis and the token server; everything else loads on demand
```

## Models

| Stage | Model | Parameters | Precision | Runs as |
|-------|-------|------------|-----------|---------|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3` | ~1.55B | INT8 (`int8_float16`) | shared service (`whisper_server.py`), GPU |
| LLM | Qwen2.5 14B Instruct (via vLLM) | 14.8B | AWQ (4-bit) | shared service (official `vllm/vllm-openai` image), GPU |
| TTS | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) | ~0.5B (T3) + S3Gen vocoder | voice-cloned from a reference clip | shared service, GPU |

All three are swappable/extensible via `config/models_config.json` without touching code — it's
the catalog the browser's model-picker reads from.

## Concurrency

The token server caps concurrent calls (`/api/token` returns 503 "line is full" past the limit)
based on the confirmed STT entry's `max_concurrent_calls` — a number that should reflect the
shared backends' real measured throughput, not a guess. faster-whisper's `num_workers` lets the
one Whisper instance handle several transcriptions in parallel; Chatterbox currently synthesizes
one request at a time behind an internal lock (concurrent requests queue, they don't corrupt each
other). No queueing at the call level — a rejected caller just retries.

## Usage

**Prerequisites:** Docker + `nvidia-container-toolkit` (GPU passthrough), a `.venv` with
`requirements.txt` installed, and `data/bank.db` seeded (`python3 src/seed_db.py`) plus the RAG
index built (`python3 src/build_index.py` — `run.sh` also does this automatically if missing).

**Quick start:**
```bash
./run.sh
```
Starts LiveKit + Redis (Docker) and the token server, then open the printed URL — pick your
LLM/STT/TTS in the browser, confirm, and the pipeline loads. `Ctrl+C` tears everything down,
including any backend containers `orchestrator.py` started.

Run the tests with `pytest tests/ -q`.
