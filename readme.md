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
Browser  ──WebRTC──▶  LiveKit SFU (box 1, Docker)
                            │  job dispatch (WebSocket)
                            ▼
                     Agent worker (src/worker.py) — box 1 only
                     — one process per active call, all workers centralized on box 1 —
                            │
              picks one pool member per call (Redis round-robin)
                            │
              ┌─────────────┼──────────────────────────────────┐
              ▼             ▼                                   ▼
            vLLM      Whisper pool                       Chatterbox pool
       (box 1 only)  (box 1: 1, box 2: 3 — 4 total)   (box 1: 1, box 2: 3 — 4 total)
```

- **LiveKit is the media server**, not this codebase — it handles the actual WebRTC
  signaling/audio transport. `src/worker.py` is a *client* of it: it opens a persistent
  connection, registers itself, and LiveKit dispatches each new call to it as a job.
- **All agent workers run on box 1, deliberately** — not spread across boxes. A worker holds no
  GPU model of its own (see next point), so running one on box 2 wouldn't add GPU capacity, and it
  would mean two independent copies of `data/bank.db`/`data/rag_index.npy` that could silently
  diverge. Centralizing on box 1 means every caller gets the same real banking/RAG data, regardless
  of which Whisper/Chatterbox pool member handles their call.
- **STT and TTS are shared HTTP microservice *pools***, not models loaded inside the call process.
  A call's job process holds no GPU model of its own — it picks one pool member (round-robin via a
  Redis counter, see Concurrency below) and makes HTTP requests to it for the call's whole
  duration. This matters for concurrency: earlier, STT was loaded fresh *per call*, which OOM'd the
  GPU on a second simultaneous caller — making it a shared, poolable service (like TTS already was)
  fixed that. See `src/whisper_server.py` / `src/whisper_stt.py`.
- **Nothing loads until you pick it.** `src/token_server.py` serves the browser's model-picker
  screen first; only after you confirm an LLM/STT/TTS choice does `src/orchestrator.py` actually
  start anything, per `config/models_config.json`.
- **The GPU-heavy backends run in Docker** (`docker-compose.yml`, `Dockerfile.whisper`,
  `Dockerfile.chatterbox`, `Dockerfile.worker`, and box 2's `docker-compose.pool.yml`) —
  `orchestrator.py` launches box 1's on demand and polls `/health` for readiness. This isolates
  their dependencies (CUDA/cuDNN versions, Python packages) from anything else on the host, and
  from other unrelated projects sharing the same GPU box (box 2 runs several).
- **This repo is fully self-contained — it doesn't build against the sibling `Pipeline/` repo.**
  `chatterbox_server.py`, `Dockerfile.chatterbox`, and `assets/voice_seed/` all live here as a
  deliberate *copy* of the originals in `Pipeline/` (the pre-LiveKit bare-metal assistant, which
  keeps its own independent copy and still uses it directly). The two are meant to be able to
  diverge — each pipeline is independently deployable without the other's directory present at
  all, at the cost of the two `chatterbox_server.py` copies needing manual sync if one is
  improved and the fix is relevant to both.
- **Each Chatterbox instance can have a different voice** (`CHATTERBOX_VOICE_FILE`, baked into the
  image from this repo's own `assets/voice_seed/`) — since a call keeps the same pool member for
  its whole duration, one caller always hears one consistent voice, but different concurrent
  callers can hear different-sounding agents.

## Domain config

All brand/company-specific data lives in `config/bank_config.json` (bank name, branches, hours,
services, contact info) — re-read on every turn, no restart needed. The bot greets automatically
the moment a call connects, using the same LLM+TTS pipeline as any other turn.

## Repository layout

```
src/            worker.py (LiveKit agent — STT/LLM/TTS orchestration per call, round-robin pool
                selection via _pick_pool_url),
                token_server.py (auth + model-picker + static web/ host),
                orchestrator.py (launches/tears down box 1's backends via Docker Compose),
                whisper_server.py + whisper_stt.py (shared STT service + its client plugin),
                chatterbox_server.py + chatterbox_tts.py (shared TTS service + its client plugin —
                both live here; chatterbox_server.py is a deliberate copy of the one in the
                sibling Pipeline/ repo, see Architecture),
                banking.py, db.py, rag.py, convo_log.py, build_index.py, seed_db.py, show_db.py
web/            Browser frontend (livekit-client) — index.html, vendor/
config/         bank_config.json (swappable domain config), models_config.json (backend
                catalog — id/label/launch params per LLM/STT/TTS option, plus extra_pool_urls for
                box 2's Whisper/Chatterbox pool members), rag_docs/
data/           bank.db, memory.json, rag_index.npy, fastembed_cache/ — gitignored,
                regenerable/mutable, box 1 only (see Concurrency: why workers stay on box 1)
assets/voice_seed/  Chatterbox reference clips (.wav) — CHATTERBOX_VOICE_FILE picks one per pool
                instance at build/run time
docker-compose.yml, docker-compose.pool.yml, Dockerfile.whisper, Dockerfile.chatterbox,
Dockerfile.worker
                Container definitions — docker-compose.yml is box 1 (see Architecture above);
                docker-compose.pool.yml is box 2's standalone Whisper/Chatterbox pool (deployed via
                ../sync-to-box2.sh into ~/box2-pool-node/ there, not part of this checkout's tree).
                All build contexts resolve within this repo — no cross-repo dependency on Pipeline/
livekit.yaml    Self-hosted LiveKit server config
tests/          pytest suite (banking/RAG logic)
plans/          Design/planning docs
logs/           Runtime logs, plus each backend's Docker container log — gitignored
run.sh          Starts LiveKit+Redis and the token server on box 1; everything else loads on demand
```
`../sync-to-PC.sh`, `../sync-to-box2.sh`, `../start-pipeline.sh`, `../stop-pipeline.sh` (one level
up, alongside this repo's sibling `Pipeline/`) are the operator scripts — see Usage below.

## Models

| Stage | Model | Parameters | Precision | Runs as |
|-------|-------|------------|-----------|---------|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3` | ~1.55B | INT8 (`int8_float16`) | pool of 4 (box 1: 1, box 2: 3), GPU |
| LLM | Qwen2.5 14B Instruct (via vLLM) | 14.8B | AWQ (4-bit) | single instance, box 1 only, GPU |
| TTS | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) | ~0.5B (T3) + S3Gen vocoder | voice-cloned per instance | pool of 4 (box 1: 1, box 2: 3), GPU |

All three are swappable/extensible via `config/models_config.json` without touching code — it's
the catalog the browser's model-picker reads from. Pool members beyond box 1's own instance are
listed per-entry under `extra_pool_urls`.

## Concurrency

**Pool sizes today: 4 Whisper instances, 4 Chatterbox instances** (1 of each on box 1, 3 of each
on box 2's `docker-compose.pool.yml`) — measured live at ~2.17GB/Whisper instance and
~3.5GB/Chatterbox instance on box 2's RTX 3090 (24.5GB), so box 2 alone is close to that card's
real ceiling already; growing meaningfully past ~4 of each per box needs another GPU machine, not
just a config change.

**Pool sizes aren't additive, but sharing a pool member is fine at this scale.** Every call needs
one Whisper instance *and* one Chatterbox instance at once — only 4 calls get a dedicated,
uncontended instance of each; beyond that, calls share. **The admission cap is 12** (past that,
`/api/token` returns 503, "line is full" — no queueing, a rejected caller just retries), which
means up to 3 calls can share one pool instance at once. This works in practice because GPU
*compute* utilization on Whisper/Chatterbox is spiky, not sustained — near-zero between turns, and
only spikes for the sub-second burst of an actual transcription or synthesis call — so 2-3 calls
interleaving their brief bursts on one instance mostly just queues microseconds of GPU work, not
whole-turn latency. This is a different resource than the ~2-3.5GB *memory* each instance holds
constantly regardless of how many calls share it.

**Selection is round-robin via a Redis atomic counter** (`src/worker.py`'s `_pick_pool_url`,
called once per call in `entrypoint()` — not per request), not random: with N simultaneous new
calls and pool size ≥ N, each call deterministically lands on a *different* instance, confirmed
live via 4 concurrent test calls hitting all 4 Whisper and all 4 Chatterbox instances with zero
repeats. Past pool size, the counter wraps and spreads extra calls evenly rather than piling them
onto whichever instance happens to be least busy. A call keeps its picked instance for its whole
duration.

**Box 2 pool members are health-checked once, at worker launch — not per call.**
`orchestrator.py`'s `_launch_worker` builds `WHISPER_URLS`/`CHATTERBOX_URLS` by hitting each
`extra_pool_urls` entry's `<host>:<port>/health` before including it; box 1's own local instance is
always included unconditionally (already proven up earlier in the same launch). Any box 2 member
that doesn't respond gets dropped from the list the worker ever sees, so if box 2's pool is down
when the pipeline starts, every call just uses box 1 alone — no dead URLs in the round-robin
rotation. This was a real bug, not a hypothetical: round-robin itself has no failover, so with box
2 stopped mid-session while still listed, calls kept getting routed to it and failed outright (mic
audio never reaching Whisper, agent replies never getting synthesized) while the call still visibly
"connected" (LiveKit's room + the text greeting don't touch STT/TTS, so nothing on screen indicated
a problem). **Known remaining gap**: this check is launch-time only — a pool member that dies while
the worker is already running isn't detected until you relaunch.

**There is exactly one worker, running on box 1 — not one per box.** Box 2 is pool-only (see
Architecture): once Whisper/Chatterbox moved out of the worker process, a worker on box 2 would
add no GPU capacity, only a second, divergeable copy of `data/bank.db`/`rag_index.npy`. Every call
routes through this single worker regardless of which pool member ends up handling its STT/TTS.

**`num_idle_processes=12` in `WorkerOptions`** (`src/worker.py`) matches the admission cap 1:1, not
a coincidence — it's how many job-executor *processes* (LiveKit's own multiprocessing under this
one worker registration) are kept pre-warmed for zero-latency dispatch. Sizing it to 12 means every
admitted caller gets an already-warm process, no cold-spawn delay for anyone up to the cap. A
second independent worker was considered and deliberately rejected: LiveKit's multiprocessing
already gives you as many concurrent job-executor processes as needed under one worker, so a second
worker (still on box 1 — box 2 isn't an option, see above) would just double registration/
load-threshold overhead and add SQLite write-contention risk between two fully independent
processes, without unlocking capacity a single worker's own process pool can't already provide.
Each idle process holds its own prewarmed RAG/VAD state in RAM, so 12 idle processes is a real,
constant CPU/RAM cost on box 1 — whether it comfortably sustains that, and how vLLM's own latency
holds up under 12-way concurrent generation, hasn't been load-tested yet.

## Usage

**Prerequisites (each GPU box):** Docker + `nvidia-container-toolkit` (GPU passthrough — verify
with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` before assuming it
works). Box 1 additionally needs a `.venv` with `requirements.txt` installed (`token_server.py`
runs bare-metal there, not in Docker), `data/bank.db` seeded (`python3 src/seed_db.py`), and the
RAG index built (`python3 src/build_index.py` — `run.sh` also does this automatically if missing).

### From a dev machine with SSH access to both boxes (typical case)

`../start-pipeline.sh` / `../stop-pipeline.sh` (one level up from this repo, alongside the sync
scripts) drive both boxes over SSH — see their own comments for the exact mechanism. They assume
`cognimind`/`cognimind2`-style SSH host aliases are already set up and that `../sync-to-PC.sh` /
`../sync-to-box2.sh` have been run at least once (code deployed, images built).

```bash
./start-pipeline.sh        # box 1 only: vLLM + 1 Whisper + 1 Chatterbox + the worker
./start-pipeline.sh -a     # also brings up box 2's pool (3 more Whisper + 3 more Chatterbox)
./stop-pipeline.sh         # reverse — stop box 1 (graceful SIGINT to run.sh, same as Ctrl+C)
./stop-pipeline.sh -a      # also stop box 2's pool
```
Both are idempotent (safe to re-run; only start/report what isn't already up) and wait for real
readiness rather than just firing requests and exiting.

### Directly on box 1 (no dev machine / no SSH in the loop)

```bash
cd "Livekit Pipeline" && ./run.sh
```
Starts LiveKit + Redis (Docker) and the token server, then open the printed URL — pick your
LLM/STT/TTS in the browser, confirm, and the pipeline loads. `Ctrl+C` tears everything down,
including any backend containers `orchestrator.py` started. This is what `start-pipeline.sh` runs
remotely on your behalf — running it locally at box 1's own terminal is exactly equivalent.

### Setting up from scratch with physical access only (no SSH between the boxes)

If you're sitting at each machine directly rather than working from a dev machine with SSH to
both — e.g. first-time setup, or SSH access genuinely isn't available — the scripts above don't
apply, but the underlying steps are simple manual copies:

**On box 1** (the LiveKit server + vLLM + the agent worker + its own Whisper/Chatterbox instance):
1. Copy this `Livekit Pipeline/` directory onto box 1 (USB drive, local network share, `git
   clone` — anything that isn't SSH from elsewhere). It's self-contained — no need to also copy
   the sibling `Pipeline/` repo unless you're separately setting up the original bare-metal
   assistant there too, which this guide doesn't cover.
2. Follow the *Prerequisites* above on that machine, then `cd "Livekit Pipeline" && ./run.sh`.
3. Open `http://localhost:3000` in a browser **on box 1 itself** (or box 1's own LAN/Tailscale
   address from another device on the same network), pick models, confirm.

**On box 2** (just the Whisper/Chatterbox pool — no LiveKit, no worker, no `config/`/`data/`):
1. Copy only what's needed to build the two pool images, preserving this exact relative layout —
   `docker-compose.pool.yml` at the top level, with `Livekit Pipeline/` as a sibling beneath it
   (mirrors what `../sync-to-box2.sh` pushes over SSH; see that script's own comments for the
   precise file list — `Dockerfile.whisper` + `Dockerfile.chatterbox` + `requirements.txt` +
   `requirements-chatterbox.txt` + `src/whisper_server.py` + `src/chatterbox_server.py` +
   `assets/voice_seed/*.wav`, all under `Livekit Pipeline/`). No `Pipeline/` directory needed on
   box 2 at all.
2. Follow the *Prerequisites* above on that machine, then from the directory containing
   `docker-compose.pool.yml`: `docker compose -f docker-compose.pool.yml up -d`.
3. **Back on box 1**, edit `config/models_config.json`'s `stt`/`tts` entries — `extra_pool_urls`
   needs box 2's *actual* LAN IP (defaults are hardcoded to this project's specific box 2; find
   yours with `ip addr` on box 2 and update both lists to match, keeping the same ports:
   `8768`-`8770` for Whisper, `8771`-`8773` for Chatterbox). Box 2 itself needs no configuration
   pointing back at box 1 — the pool services are just plain HTTP servers with no awareness of who
   calls them.
4. Restart box 1's pipeline (`./run.sh` again, or just re-confirm the model selection in the
   browser if it's already running) so the worker picks up the updated pool list.

Run the tests with `pytest tests/ -q`.
