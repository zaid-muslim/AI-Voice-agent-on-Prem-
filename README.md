# Voice Agent Pipeline

A real-time, fully **on-premises** voice assistant. A caller speaks in their browser; the system
transcribes the speech, generates a grounded reply with a local language model, and speaks back in
a cloned voice — every stage running on hardware you control, with no audio, transcript, or
customer data ever leaving your network.

The reference deployment answers as a retail-bank branch receptionist: it answers informational
questions (branches, hours, services, contact details) and detailed product/fee/rate/policy
questions grounded in its own documents, blocks a lost or stolen card after verifying the caller's
identity, and logs a callback request for a human. It cannot check balances or move money — that
boundary is enforced in code, not just the prompt.

**What "on-premises" means here:**

- **No cloud, no per-call fees, no external dependency in the inference path.** Speech recognition
  (Whisper), the language model (Qwen 2.5 via vLLM), and voice synthesis (Chatterbox) all run
  locally on your own GPUs.
- **Self-hosted media server.** The WebRTC audio transport ([LiveKit](https://livekit.io/)) runs on
  your LAN too — call audio never touches the public internet.
- **Regulator-friendly.** Because customer speech and data stay inside your network, the system
  suits banking and other settings where data cannot be sent to a third-party provider.

The stack scales across two GPU machines: **box 1** runs the full pipeline on its own, and **box 2**
optionally adds speech-processing capacity for more concurrent callers.

## Architecture

```
        ┌───────────┐         WebRTC audio          ┌───────────────┐
        │  Browser  │ ◀───────────────────────────▶ │  LiveKit SFU  │
        │ (web/ UI) │   loads UI + access token      │ (self-hosted) │
        └───────────┘   from the Token Server        └───────┬───────┘
                                                             │ dispatches each call as a job
                                                             ▼
                                                    ┌───────────────┐
                                    Redis ────────▶ │  Agent Worker │   one worker (box 1),
                                 (round-robin pick)  │  (per call)   │   owns the DB + RAG data
                                                    └──┬────┬────┬───┘
                                       each turn:  STT ┘    │    └ TTS
                                                        LLM │
                        ┌────────────────────────────────────┼──────────────────────────────┐
                        ▼                                     ▼                              ▼
                ┌───────────────┐                    ┌───────────────┐             ┌───────────────┐
                │  Whisper STT  │                    │  vLLM  (LLM)  │             │ Chatterbox TTS│
                │     pool      │                    │  box 1 only   │             │     pool      │
                └───────────────┘                    └───────────────┘             └───────────────┘
                box 1 ×1 + box 2 ×3                                                box 1 ×1 + box 2 ×3
```

- **LiveKit** is the self-hosted media server; the **Agent Worker** is a client that receives each
  call dispatched to it. Per conversational turn, the worker calls three local services over HTTP:
  Whisper (speech-to-text) → vLLM (the LLM) → Chatterbox (text-to-speech).
- **The Token Server** (`:3000`) serves the browser UI, issues signed LiveKit access tokens, and
  enforces the concurrent-call limit. Nothing heavy starts until you pick models in the browser;
  the **Orchestrator** then launches the Docker backends on demand.
- **One worker, on box 1**, owns all call state — the SQLite database and the RAG document index —
  so every caller reaches the same data. Box 2 adds only stateless speech-processing capacity.

### Concurrency

- **Pools:** 4 Whisper + 4 Chatterbox instances (box 1 ×1 + box 2 ×3 of each). The worker picks one
  of each per call via a **Redis round-robin** counter, spreading concurrent calls across the pool.
- **Admission cap:** 12 concurrent calls (`max_concurrent_calls` in `config/models_config.json`).
  Past that, the Token Server returns HTTP 503 ("line is full") and the caller retries — there is no
  queue. Up to 3 calls may briefly share one instance, which is fine because GPU compute per call is
  spiky (sub-second bursts), not sustained.
- **Graceful fallback:** box 2's pool members are health-checked when the worker launches; if box 2
  is down, box 1 runs on its own instances with no dead entries in the rotation.
- **Warm processes:** the single worker keeps 12 job-executor processes pre-warmed
  (`num_idle_processes`), so admitted callers get zero cold-start delay.

## Models

| Stage | Model | Parameters | Precision | Runs as |
|-------|-------|------------|-----------|---------|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3` | ~1.55B | INT8 (`int8_float16`) | pool of 4 (box 1: 1, box 2: 3), GPU |
| LLM | Qwen 2.5 14B Instruct (via vLLM) | 14.8B | AWQ (4-bit) | single instance, box 1 only, GPU |
| TTS | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) | ~0.5B (T3) + S3Gen vocoder | voice-cloned per instance | pool of 4 (box 1: 1, box 2: 3), GPU |

All three are swappable via `config/models_config.json` — the catalog the browser's model-picker
reads from — without touching code.

## Configuring for your organization

The pipeline is domain-generic: the assistant's identity, knowledge, and models are all
configuration, so the same code can serve a bank, a clinic, a utility, or any other
informational-plus-actions voice desk. Three files drive it, no code changes required:

- **`config/bank_config.json`** — the organization's identity and facts (name, locations, hours,
  services, contact info). Re-read on every turn, so edits take effect without a restart.
- **`config/rag_docs/`** — the source documents the assistant grounds its detailed answers in (fees,
  rates, policies, product details). Drop in your own documents and rebuild the index
  (`src/build_index.py`) to retarget the knowledge base.
- **`config/models_config.json`** — which LLM/STT/TTS to run, and the box-2 pool URLs.

To adapt to a new domain: replace `bank_config.json` with your organization's facts, put your
documents in `rag_docs/` and rebuild the index, and adjust the assistant's prompt/tools if the task
differs from the reference (informational Q&A and RAG are fully generic; a specialized action like
card-blocking is application logic you would tailor to your workflow).

## Repository layout

```
Livekit Pipeline/
├── src/
│   ├── worker.py               # LiveKit agent: per-call STT→LLM→TTS + round-robin pool pick
│   ├── token_server.py         # serves web/ UI, issues LiveKit tokens, enforces the call cap
│   ├── orchestrator.py         # launches/stops the Docker backends, polls readiness
│   ├── whisper_server.py       # shared Whisper STT microservice (FastAPI)
│   ├── whisper_stt.py          # STT client plugin used by the worker
│   ├── chatterbox_server.py    # shared Chatterbox TTS microservice (FastAPI)
│   ├── chatterbox_tts.py       # TTS client plugin used by the worker
│   ├── banking.py, db.py        # domain logic + SQLite access
│   ├── rag.py, convo_log.py     # document retrieval + conversation logging
│   └── seed_db.py, build_index.py, show_db.py   # setup / maintenance utilities
├── web/                        # browser frontend (LiveKit client) — index.html, vendor/
├── config/
│   ├── bank_config.json        # domain content (name, branches, hours, services)
│   ├── models_config.json      # backend catalog + box-2 pool URLs (extra_pool_urls)
│   └── rag_docs/               # source documents the assistant grounds answers in
├── assets/voice_seed/          # Chatterbox reference voice clips (.wav)
├── data/                       # bank.db, memory.json, rag_index.npy, caches (gitignored)
├── Dockerfile.whisper          # Whisper STT image
├── Dockerfile.chatterbox       # Chatterbox TTS image
├── Dockerfile.worker           # agent worker image
├── docker-compose.yml          # box 1 stack (LiveKit, Redis, vLLM, Whisper, Chatterbox, worker)
├── docker-compose.pool.yml     # box 2 pool (3× Whisper + 3× Chatterbox)
├── livekit.yaml                # self-hosted LiveKit server config (API keys, ports)
├── requirements.txt            # Python deps for the token server + worker (host, box 1)
├── requirements-chatterbox.txt # Python deps baked into the Chatterbox image
├── run.sh                      # box 1 launcher (LiveKit + Redis + token server)
└── tests/                      # pytest suite (banking / RAG logic)
```

## Setup

Do this once per machine. Both boxes need Docker with GPU passthrough; **box 1** additionally needs
the Python environment (the token server and orchestrator run on the host, not in Docker).

### 1. Prerequisites (both boxes)

- An NVIDIA GPU (RTX 3090-class) with a recent driver.
- **Docker Engine** + the **Docker Compose** plugin.
- **NVIDIA Container Toolkit** for GPU passthrough into containers. Verify it works before
  continuing:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

### 2. Clone

```bash
git clone <repo-url>
cd "Livekit Pipeline"
```

### 3. Python environment — box 1 only

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

### 4. Environment file — box 1 only

Create `.env` in the repo root. `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` must match a key pair in
`livekit.yaml` (`keys:`); the values below match the dev key shipped in this repo.

```dotenv
# LiveKit server (key/secret must match livekit.yaml `keys:`)
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=Fh/JM5WbHJ466Ir9vfmtC9jDOaCZZQVYYFFWd/G7AiA=

# Backend URLs — fallback defaults for running the worker directly. In normal operation the
# orchestrator overrides these per the model selection confirmed in the browser.
VLLM_URL=http://127.0.0.1:8000/v1
VLLM_MODEL=qwen2.5-14b-awq
WHISPER_URLS=http://127.0.0.1:8768/transcribe
CHATTERBOX_URLS=http://127.0.0.1:8766/synthesize
REDIS_URL=redis://127.0.0.1:6379
```

> For production, generate your own LiveKit secret and change it in **both** `livekit.yaml` and
> `.env`. Use `127.0.0.1`, not `localhost`, for container-facing URLs (minimal container images
> can't resolve the literal `localhost`).

### 5. Seed the database and build the RAG index — box 1 only

```bash
.venv/bin/python src/seed_db.py        # creates data/bank.db
.venv/bin/python src/build_index.py    # builds data/rag_index.npy from config/rag_docs/
```

(`run.sh` also builds the RAG index automatically if it's missing.)

### 6. Docker images

- **Box 1** — LiveKit, Redis, and vLLM use prebuilt images (pulled on first run). The Whisper,
  Chatterbox, and worker images build automatically the first time the orchestrator launches them.
  To build them ahead of time:

  ```bash
  docker compose --profile on-demand build
  ```

- **Box 2** — the two pool images build on the first `docker compose -f docker-compose.pool.yml
  up -d`. On a fresh machine, bring the instances up **one at a time** the first time, so six
  simultaneous model-weight downloads don't contend for bandwidth.

## Running

Launch each machine directly at its own terminal — no SSH required. **Box 1 alone is a complete,
working pipeline**; add box 2 only to scale the pools.

### Box 1 (main node)

From the repo root:

```bash
./run.sh
```

This starts LiveKit + Redis (Docker) and the token server, and prints a URL. Open it in a browser
— on box 1 itself, or from any device on the same LAN using box 1's IP address — choose your
LLM/STT/TTS, and confirm. The orchestrator then launches vLLM, Whisper, Chatterbox, and the worker
on demand. `Ctrl+C` tears everything down.

### Box 2 (optional pool node)

On box 2, from the directory containing `docker-compose.pool.yml`:

```bash
docker compose -f docker-compose.pool.yml up -d      # start 3× Whisper + 3× Chatterbox
docker compose -f docker-compose.pool.yml ps         # check status / health
docker compose -f docker-compose.pool.yml down       # stop
```

### Pointing box 1 at box 2

Box 1 finds box 2's pool through `config/models_config.json` — each `stt` / `tts` entry's
`extra_pool_urls`. Set these to box 2's LAN IP (find it with `ip addr` on box 2), keeping the
ports: Whisper `8768`–`8770`, Chatterbox `8771`–`8773`. Then re-confirm the model selection in the
browser so the worker picks up the pool. If box 2 is down when box 1 starts, box 1 automatically
falls back to its own instances.

### Tests

```bash
pytest tests/ -q
```
