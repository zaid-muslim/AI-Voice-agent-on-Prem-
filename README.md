# Riverside General — AI Voice Receptionist

A self-hosted AI phone receptionist for a (fictional test) hospital, built on
[LiveKit Agents](https://docs.livekit.io/agents/). A caller can book, cancel,
or reschedule an appointment, ask general questions, and speak with a
natural-sounding voice — with a deterministic, LLM-bypassing safety gate for
real medical emergencies.

This README is the single source of truth for the project (it replaces the
old `Readme_technical.md` / `Readme_nontechnical.md` pair). It's written for
whoever has to run, deploy, or extend this system next.

---

## 1. What changed in this revision

Three real problems, fixed:

1. **STT no longer scales per-room.** LiveKit spawns one worker *process*
   per concurrent call, and the old `faster-whisper` STT plugin loaded its
   own model inside every one of those processes — N concurrent callers
   meant N separate whisper models resident in GPU memory at once,
   competing with the LLM and TTS for the same card. **Fixed** by pulling
   STT out into [`stt_service/`](stt_service/server.py): one persistent,
   GPU-warm `faster-whisper` (CTranslate2) process, shared by every
   worker/room over HTTP, using CTranslate2's own `num_workers` replica
   pool for concurrency instead of spawning more model copies — the same
   "one warm shared server" pattern already used for the LLM (vLLM) and TTS
   (vLLM-Omni on PC2). See [§4](#4-architecture).
2. **No way to add a doctor's timings without touching the database by
   hand.** Fixed with a full doctor-schedule admin surface — add a new
   doctor (with an initial weekly schedule in the same form), add
   individual or recurring timings for an existing doctor, and remove
   unbooked slots — see [§6](#6-managing-doctors--schedules).
3. **The project wasn't organized, had no single dependency manifest, and
   wasn't dockerized.** Fixed — see [§2](#2-repository-layout) and
   [§7](#7-docker-deployment).

---

## 2. Repository layout

```
voice-agent-pipeline/
├── app/                      # the agent + its FastAPI control surfaces
│   ├── main.py                LiveKit worker entrypoint (agent persona,
│   │                          function tools, safety gate, session wiring)
│   ├── token_server.py         mints join tokens + serves frontend/ (:7860)
│   ├── admin_server.py         hospital data + doctor/schedule admin (:7870)
│   ├── dev_server.py           model/backend switcher, latency compare (:7871)
│   ├── system_config.py/.json  which LLM/STT/TTS is active right now
│   ├── models_registry.json    curated model/engine catalogue for the dev UI
│   ├── hospital_core/          booking.py (SQLite), hospital_kb.py, rag.py,
│   │                          safety.py - the domain logic
│   ├── plugins/                STT/TTS engine adapters (see §5)
│   ├── frontend/index.html     caller-facing web UI
│   ├── admin_ui.html            hospital data + doctor schedule admin UI
│   ├── requirements.txt
│   └── .env                    local config (see §3)
├── stt_service/               shared faster-whisper STT server (see §4)
├── remote/qwen_tts_server/    PC2 (TTS box) Dockerfile + load test
├── docker/                    all Dockerfiles + compose files (see §7)
├── scripts/run_vllm.sh        vLLM launch script (bare-metal, non-Docker)
├── chat_templates/            Gemma tool-calling chat template
├── tests/test_booking.py      real pytest coverage for hospital_core/booking.py
└── models/, archive/, remote_services/, data/   gitignored - see below
```

**Deliberately gitignored / not committed:**
- `models/` — model weights, fetched separately (see §8).
- `archive/` — a dead-end STT experiment (a standalone Nemotron ASR server)
  kept on disk for reference, superseded by `stt_service/`.
- `remote_services/` — a local vendored clone of `vllm-project/vllm-omni`
  (its own git history) used during development; `remote/qwen_tts_server/`
  is the real, tracked, minimal PC2 setup that installs the same project
  from its own repo instead of vendoring it.
- `data/`, stray `*.wav`/`*.db` files — scratch/runtime output, not source.
- The 7 leftover per-engine virtualenvs at the repo root (`.venv-parakeet`,
  `.venv-kokoro`, etc.) — see [§5](#5-stt--tts-engines) for why they exist.

---

## 3. Configuration

`app/.env` holds machine-specific config (paths, IPs, credentials — never
commit this file). `app/system_config.json` holds *which* LLM/STT/TTS engine
is active right now, edited live via the dev console at `:7871` or by hand —
see that file's docstring for the exact "takes effect on the next call, not
this one" contract.

Key `.env` variables:

```bash
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_WS_URL=ws://<this-machine's-LAN-IP>:7880   # what the BROWSER dials

VLLM_BASE_URL=http://127.0.0.1:8000/v1

# Shared STT service (stt_service/server.py) - the default engine
SHARED_STT_BASE_URL=http://localhost:8020

# Qwen3-TTS via vLLM-Omni, running persistently on PC2 (see §4)
QWEN_OMNI_BASE_URL=http://<PC2-LAN-IP>:8091/v1
QWEN_OMNI_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_OMNI_VOICE=Aiden

ADMIN_PASSWORD=change-me
DEV_PASSWORD=change-me
```

---

## 4. Architecture

```
Caller's browser/phone
        │
        ▼
livekit-server (signaling + media, self-hosted)
        │
        ▼
app/main.py ── AgentSession(vad, stt, llm, tts, turn_detection)
        │             │        │      │
        │             │        │      └─ HTTP → PC2's vLLM-Omni (Qwen3-TTS)
        │             │        └─ HTTP → vLLM (Gemma 4 12B)
        │             └─ HTTP → stt_service/server.py (shared faster-whisper)
        ▼
hospital_core/ (booking.py, safety.py, rag.py, hospital_kb.py)
```

Three persistent, shared, GPU-warm servers — **not** spawned per call — sit
behind the agent: vLLM for the LLM, `stt_service/server.py` for STT, and
vLLM-Omni (on a second machine, "PC2") for TTS. Every concurrent room hits
the *same* three servers over HTTP instead of each getting its own copy;
this is the one architectural idea the whole stack is built around, and
`stt_service/` brings STT in line with the other two.

**Three web interfaces:**

| Interface | Port | File | Purpose |
|---|---|---|---|
| Caller | 7860 | `app/token_server.py` | The person phoning in |
| Hospital admin | 7870 | `app/admin_server.py` | Doctors, schedules, hours |
| Developer console | 7871 | `app/dev_server.py` | Model switching, latency comparison |

**Why TTS lives on a second machine ("PC2")**: freeing PC1's GPU budget for
STT + LLM, while PC2's TTS server stays warm and shared across every caller.
STT does **not** need a second machine to get the same benefit — a shared
server on the *same* box already removes the per-process multiplication
problem, since the bottleneck was redundant model copies, not raw GPU
contention with the LLM (whisper's memory footprint is small next to
Gemma's). Split `stt_service/` onto its own machine later the same way, via
`SHARED_STT_BASE_URL`, if STT and LLM ever do start contending for VRAM.

---

## 5. STT / TTS engines

**Default, production path:**
- STT: shared `faster-whisper distil-large-v3` via `stt_service/server.py`
  (engine `whisper_shared` in `system_config.json`). Falls back
  automatically to the in-process `plugins/whisper_stt.py` (engine
  `whisper`) if the shared service is unreachable at worker startup.
- TTS: Qwen3-TTS via vLLM-Omni on PC2 (engine `qwen_omni`, the default).
  Falls back to a local subprocess bridge (`plugins/qwen_tts.py`, engine
  `qwen_local_subprocess`) if PC2 is unreachable.

**Candidate engines** (real, working, but each needs its own isolated
virtualenv on the host and are **not** included in the Docker images —
dockerizing five extra heavyweight ML stacks that aren't the actual
production path wasn't worth the image bloat/build complexity):

| Engine | Type | Needs | License note |
|---|---|---|---|
| Parakeet TDT (STT) | subprocess | `.venv-parakeet`, own NeMo build | CC-BY-4.0 |
| Canary 180M Flash (STT) | subprocess | same NeMo venv as Parakeet | CC-BY-4.0 (use the *flash* variant, not `canary-1b`, which is non-commercial) |
| Chatterbox (TTS) | subprocess | `.venv-chatterbox`, pins `torch==2.6.0` | see resemble-ai/chatterbox |
| Kokoro-82M (TTS) | subprocess | `.venv-kokoro` | Apache-2.0 |
| Piper (TTS) | subprocess, CPU-only | `.venv-piper` | **GPL-3.0-or-later** — get legal sign-off before commercial use |

Each is selected via `system_config.json`'s `stt`/`tts.engine`, and each
falls back to the production default if its required `*_PYTHON`/`*_WORKER`
env vars aren't set — see `app/main.py`'s `_make_stt()`/`_make_tts()`.

---

## 6. Managing doctors & schedules

`app/admin_ui.html` (served at `:7870`, password-protected) has two tabs:

- **Hospital Data** — doctor bios, departments, hours, and other
  informational content the AI quotes to callers (`hospital_data.json`).
- **Doctor Schedules** (new) — the actual bookable calendar
  (`hospital_core/booking.py`'s SQLite `slots` table):
  - **Add a new doctor**: name, department, an optional bio, and
    (optionally, in the same form) an initial recurring weekly schedule —
    e.g. "Mon/Wed/Fri, 09:00–13:00, 30-minute slots, next 4 weeks." Writes
    the bio into hospital data AND the initial timings in one call, closing
    the "two sources of truth must be kept in sync by hand" gap the old
    code only warned about in a comment.
  - **Add timings for an existing doctor**: either explicit one-off
    date/time slots, or the same recurring weekly pattern.
  - **Remove a timing**: only for slots nobody has booked yet — a booked
    slot must be cancelled through the normal call flow first.

Underlying API (`app/admin_server.py`, all behind the same admin
Basic-Auth): `GET /api/doctors`, `GET /api/doctors/slots`,
`POST /api/doctors`, `POST /api/doctors/slots`, `DELETE /api/doctors/slots`.

---

## 7. Docker deployment

Everything runs in containers except the two per-machine model directories
(`models/` and PC2's `models/qwen3-tts/`), which are volume-mounted rather
than baked into images. Requires the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on both machines (`docker info | grep -i runtime` should list `nvidia`).

**PC1** (agent + LLM + STT + the three web interfaces):

```bash
cp app/.env.example app/.env   # if you don't already have one - edit for your machine
docker compose -f docker/docker-compose.yml up -d --build
```

Services: `livekit-server` (:7880), `vllm` (:8000, Gemma 4 12B),
`stt-service` (:8020, shared faster-whisper), `agent` (the LiveKit worker,
no exposed port), `token-server` (:7860), `admin-server` (:7870),
`dev-server` (:7871). `hospital_core/` is bind-mounted (not baked into the
image) so `agent` and `admin-server` share one real, persistent database.

**PC2** (Qwen3-TTS box) — copy `remote/qwen_tts_server/` and
`models/qwen3-tts/` to the second machine, then:

```bash
docker compose -f docker/docker-compose.pc2.yml up -d --build
```

Point PC1's `QWEN_OMNI_BASE_URL` at this machine's LAN IP, port 8091.

**Known Docker-specific limitation**: the `qwen_local_subprocess` /
candidate-engine fallbacks that spawn a subprocess in a separate host venv
(`.venv-voice`, `.venv-parakeet`, etc.) aren't available inside the
container, since those venvs don't exist there. If PC2 is unreachable, the
containerized `agent` will log the failure rather than silently falling
back to a local subprocess — run bare-metal (not via Docker) instead if you
need that specific fallback path available.

---

## 8. Bare-metal setup (no Docker)

**PC1:**
```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r app/requirements.txt
cd app && python main.py download-files
```
`livekit.yaml` must bind `0.0.0.0`, not just loopback, or nothing outside
`localhost` can connect. Then, one terminal each:
```bash
livekit-server --config app/livekit.yaml --dev
bash scripts/run_vllm.sh                              # LLM
python stt_service/server.py                          # shared STT
cd app && python main.py dev                           # agent
uvicorn token_server:app --host 0.0.0.0 --port 7860
uvicorn admin_server:app --host 0.0.0.0 --port 7870
uvicorn dev_server:app   --host 0.0.0.0 --port 7871
```

**PC2:**
```bash
pip install git+https://github.com/vllm-project/vllm-omni.git
vllm serve /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8091
```

---

## 9. What's proven vs. what's a candidate

Not every engine in `models_registry.json` has been used on a real, live
call. Each entry is labeled **verified** (actually run on this project's own
hardware/logs) or **candidate** (a real, working integration, API-checked
against its published source, but not yet proven end-to-end here) — check
that file before trusting an option in production.

---

## 10. Testing

```bash
pip install pytest
pytest tests/test_booking.py -v
```

Covers the real booking engine: double-booking rejection (sequential *and*
concurrent), fuzzy department/doctor-name matching, cancel/reschedule flows,
the past-time slot filter, and the new schedule-management functions
(recurring-schedule generation, slot removal, refusing to remove a booked
slot).

---

## 11. Capacity (measured on a single RTX 3090, 24GB)

- vLLM (Gemma 4 12B, bfloat16 KV cache — this GPU has no native FP8
  hardware): `--max-num-seqs 4`, tuned for **3 concurrent callers** with
  headroom, ~9.6GB reserved at `--gpu-memory-utilization 0.50`.
- Marginal cost per concurrent caller (STT + local TTS fallback, not vLLM):
  ~4.6GB with Qwen-local, ~2.7GB with Kokoro.
- Offloading TTS to PC2 frees additional PC1 budget for concurrent
  STT+LLM callers beyond the 3-caller baseline above (not yet re-measured
  post-split).

See `scripts/run_vllm.sh`'s comments for the full memory-budget derivation,
including the real vLLM startup failure that led to `--max-model-len 4096`.
