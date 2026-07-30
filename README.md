# Riverside General — AI Voice Receptionist

A self-hosted, real-time voice AI receptionist for a (fictional test)
hospital, built on [LiveKit Agents](https://docs.livekit.io/agents/). A
caller talks to it over a phone or a browser tab: it checks availability,
books, cancels, and reschedules appointments; answers general hospital
questions; and deterministically escalates real medical emergencies — all
in natural, low-latency speech, with every model in the pipeline running
on infrastructure you control. No cloud LLM, no cloud STT/TTS, no cloud
turn-detection API — the entire voice pipeline is self-hosted.

The project ships **two independent voice pipelines** behind the same
caller-facing frontend:

| Pipeline | Port | Status | What it does |
|---|---|---|---|
| **Cascade** (`app/`) | `:7860` | Production | Whisper STT → text-only Gemma 4 12B LLM → TTS |
| **Direct-audio** (`direct_audio_agent/`) | `:7862` | Experimental | Caller audio → Gemma 4 12B directly (no separate STT model) → TTS |

This document is the single source of truth for the project: what it is,
how it's built, and how to run it. A deeper, presentation-oriented
history of every design decision, bug, and measurement taken while
building the direct-audio pipeline lives in `PROJECT_PRESENTATION.md`, if
you want the full story rather than the production reference.

---

## Table of contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Voice pipeline & responsiveness](#3-voice-pipeline--responsiveness)
4. [Configuration & reliability](#4-configuration--reliability)
5. [STT / TTS engines](#5-stt--tts-engines)
6. [The direct-audio pipeline](#6-the-direct-audio-pipeline)
7. [Managing doctors & schedules](#7-managing-doctors--schedules)
8. [Deployment](#8-deployment)
9. [Testing](#9-testing)
10. [Capacity & performance](#10-capacity--performance)
11. [Troubleshooting](#11-troubleshooting)
12. [Repository layout](#12-repository-layout)

---

## 1. Overview

The cascade agent is a single LiveKit Agents worker (`app/main.py`)
sitting on an `AgentSession` pipeline — **VAD → semantic turn detection →
STT → LLM → TTS** — wired to a real hospital domain layer
(`hospital_core/`) instead of a toy demo backend:

- **Booking** — check availability, book, cancel, and reschedule
  appointments against a real SQLite-backed schedule
  (`hospital_core/booking.py`), with fuzzy department/doctor-name
  matching and double-booking rejection under concurrent requests.
- **General Q&A** — hours, departments, doctor bios, insurance, billing,
  parking, lab results, visiting policy — answered via a small RAG layer
  over hospital data (`hospital_core/rag.py`, `hospital_kb.py`), with a
  keyword-search fallback if the embedding model isn't installed.
- **Safety gate** — a fast, deterministic pattern match
  (`hospital_core/safety.py`) runs on every transcript *before* the LLM
  ever sees it. A match (e.g. "I'm having chest pain") speaks a fixed
  escalation message immediately and raises `StopResponse()` — the LLM
  never generates for that turn, so no prompt can be talked around it.
  This is defense-in-depth, not the only safety layer: the LLM's own
  system prompt is also instructed to treat ambiguous distress carefully.
- **Conversation memory that survives long calls** — every booking/lookup
  tool call folds what the caller just told it (name, department, doctor,
  date, time) into the agent's own instructions, not just the raw
  back-and-forth transcript — see [§4](#4-configuration--reliability) for
  why this matters and what real failure it prevents.

The **direct-audio pipeline** (`direct_audio_agent/`) reuses this exact
persona, tool set, and safety gate unmodified, but replaces the
STT-then-text-LLM shape with a single audio-native Gemma call for the
real reply — see [§6](#6-the-direct-audio-pipeline).

Three **web interfaces** sit alongside both pipelines:

| Interface | Port | File | Purpose |
|---|---|---|---|
| Caller (cascade) | 7860 | `app/token_server.py` | The person phoning/browsing in |
| Caller (direct-audio) | 7862 | `direct_audio_agent/call_server.py` | Same frontend, different worker |
| Hospital admin | 7870 | `app/admin_server.py` | Doctors, schedules, hours (password-gated) |
| Developer console | 7871 | `app/dev_server.py` | Model switching, live latency comparison (password-gated) |

---

## 2. Architecture

```
Caller's browser/phone
        │
        ▼
livekit-server (signaling + media, self-hosted)
        │
        ├──────────────────────────────┬──────────────────────────────┐
        ▼                              ▼                              ▼
app/main.py                    direct_audio_agent/agent.py     (both reuse)
AgentSession(vad, stt,          AgentSession(vad,               hospital_core/
  llm, tts, turn_handling)       stt=GemmaDirectAudioSTT,        (booking.py,
  │      │        │              llm=GemmaDirectAudioLLM,        safety.py,
  │      │        └─ TTS         tts, turn_handling)              rag.py,
  │      └─ vLLM (Gemma 4 12B)   │                                hospital_kb.py)
  └─ shared faster-whisper       └─ vLLM (Gemma 4 12B, audio-conditioned)
```

Persistent, GPU-warm servers sit behind both agents — **not** spawned per
call: vLLM for the LLM (shared by both pipelines), `stt_service/server.py`
for STT (cascade only), and a TTS engine (remote on a second machine, or a
fast local option — see §5). Every concurrent room hits the *same*
servers over HTTP instead of each getting its own model copy. This is the
single architectural idea the whole stack is built around, for two
reasons at once:

1. **Speed** — a cold model load (seconds) never happens on a real
   caller's turn; everything is warmed once at process start (see
   [§3](#3-voice-pipeline--responsiveness)).
2. **Memory** — LiveKit spawns one worker *process* per concurrent call.
   Without this pattern, N concurrent callers would mean N separate
   copies of every model resident in GPU memory at once, competing for
   the same card.

**Why the two pipelines can safely run side by side**: the direct-audio
worker uses LiveKit's *explicit dispatch* (`agent_name=
"direct-audio-receptionist"`) — a room only reaches it if a caller's
token specifically requests that name. The cascade's worker auto-dispatches,
unchanged. Neither can ever steal the other's room, even running
simultaneously — though running both *does* compete for the same GPU; see
[§10](#10-capacity--performance) for the real measurement behind why they
default to separate deployment profiles instead.

**Why TTS can live on a second machine ("PC2")**: frees PC1's GPU budget
for STT + LLM, while PC2's TTS server stays warm and shared across every
caller. This is optional — §5 covers a fast local alternative that needs
no second machine at all.

---

## 3. Voice pipeline & responsiveness

The pipeline is tuned around one goal: **the agent should feel like it's
actually listening, not processing** — minimizing the gap between the
caller finishing a thought and the agent's reply starting. In practice,
once the agent is confident you've finished speaking, it typically starts
responding in **well under a second**, thanks to the combination below.
Every technique here is real and running in `app/main.py` (and, where
noted, `direct_audio_agent/agent.py`), verified against the actual
running system, not aspirational:

**Semantic turn detection, not a fixed silence timer.** A fixed timer
("0.3s of quiet = they're done") is the classic failure mode: too short
and the agent barges in mid-thought; too long and every reply feels
sluggish. This pipeline splits the question in two — Silero VAD answers
"is there sound," while a semantic turn-detector model
(`livekit.agents.inference.TurnDetector`) answers "is the *thought*
actually finished." It's pinned to run **fully locally**
(`version="v1-mini"`) rather than LiveKit's cloud-hosted default — this
stack has no other cloud dependency, and a semantic decision on the
hottest path in the system shouldn't add a network hop. It's also
audio-native: it classifies directly on the caller's audio stream, not on
a finished transcript, so it doesn't wait on STT to decide whether the
turn is over — a property the direct-audio pipeline's single-call design
depends on directly (§6).

**Adaptive endpointing, tuned against real calls, not a guess.** The
wait after the turn detector signals "likely done" adapts to the
caller's own recent speaking cadence (an EMA over recent turns), bounded
by a floor and ceiling, both tuned per-pipeline against real call data —
see §10 for the numbers behind each pipeline's specific tuning.

**Preemptive generation.** The LLM starts inferring on stable partial
transcript text *before* the turn is even confirmed finished — by the
time the turn detector confirms end-of-turn, the LLM may already be
partway through its answer.

**Streaming end to end.** LLM tokens stream from vLLM as they're
generated, and TTS audio streams back progressively rather than waiting
for a full reply to synthesize before any of it plays. STT is the one
stage that stays batch rather than streaming in the cascade, by
deliberate choice: `faster-whisper` is a batch-decode model, and it isn't
the bottleneck in that pipeline anyway.

**Nothing cold on a caller's first turn.** VAD, STT/LLM, and the RAG
embedder are all warmed once per worker process *before* that process
accepts any job; the LLM and TTS both get a real warm-up call at session
start.

**An immediate acknowledgment on the turns that genuinely need it.**
Some requests (particularly ones that need a tool call) take a moment
longer than others. Rather than leaving the caller in silence, the agent
speaks a short, natural acknowledgment ("One moment," "Let me check that
for you") *only* when a reply is taking longer than expected, tuned
against real call data so it doesn't fire on turns that were about to
answer anyway. The cascade pre-renders this filler audio at startup
(bypassing live TTS synthesis contention on the hot path); falls back to
live synthesis if pre-rendering wasn't available for a given phrase.

**Measure, don't guess.** `latency_log.py` + the dev console (`:7871`)
record real per-turn timing for every cascade call, tagged by which
LLM/STT/TTS combination was active. `tests/manual/` holds the actual
measurement tools; `direct_audio_agent/benchmark.py` is the equivalent
for the direct-audio pipeline (§6, §10).

---

## 4. Configuration & reliability

`app/.env` holds machine-specific config (paths, IPs, credentials — never
commit this file; see `app/.env.example`). `app/system_config.json` holds
*which* LLM/STT/TTS engine is active right now, edited live via the dev
console at `:7871` or by hand — takes effect on the next call, not the
current one. Both pipelines read the same file.

### Long conversations don't break

Two real problems, found on live calls and fixed properly:

1. **Chat history is bounded against the LLM's context window.** A
   sufficiently long call could otherwise cross vLLM's configured context
   limit and fail identically on *every* subsequent turn — the agent
   would just stop responding, permanently, for that call. Fixed via
   `ChatContext.truncate()` in `on_user_turn_completed` — the framework's
   own sanctioned tool for this: keeps recent history, always preserves
   the system prompt, never leaves a dangling tool call without its
   result.
2. **Key facts survive even when old turns get trimmed.** Every
   booking/lookup tool call folds whatever the caller just stated into
   the agent's own instructions (via `update_instructions()`, rebuilt
   from the base prompt each time so a corrected date replaces the old
   one instead of both lingering) — so a long conversation never
   "forgets" who it's talking to.

Key `.env` variables:

```bash
# REQUIRED, no default - generate with:
#   docker run --rm livekit/livekit-server generate-keys
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
LIVEKIT_KEYS="<key>: <secret>"        # same pair, format livekit-server reads

VLLM_BASE_URL=http://127.0.0.1:8000/v1
SHARED_STT_BASE_URL=http://localhost:8020
QWEN_OMNI_BASE_URL=http://<PC2-LAN-IP>:8091/v1
QWEN_OMNI_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_OMNI_VOICE=Aiden

EMERGENCY_NUMBER=1122                 # set for YOUR deployment region
ADMIN_PASSWORD=change-me
DEV_PASSWORD=change-me
```

**Fail-closed by design**: `main.py` and `token_server.py` both refuse to
start if `LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET` are unset *or* still equal
LiveKit's well-known `--dev` default (`devkey`/`secret`), so a deployment
that forgot to override the public dev default fails loudly at startup
instead of silently handing out forgeable tokens.

### Security considerations

- **Admin (`:7870`) and dev (`:7871`) consoles** use HTTP Basic Auth,
  which sends credentials base64-encoded on every request — fine on a
  trusted LAN, not fine over the open internet without a TLS-terminating
  reverse proxy in front (none is included in this repo).
- **`livekit-server --dev`** (used in `docker/docker-compose.yml`) is a
  deliberate choice for this project's single-box, non-clustered
  deployment model — see LiveKit's own
  [clustering docs](https://docs.livekit.io/transport/self-hosting/deployment/)
  if you outgrow one machine.
- **`EMERGENCY_NUMBER`** defaults to Pakistan's "1122" — a placeholder,
  not a universal number. `hospital_core/safety.py` is explicit that this
  is a heuristic prototype, not a validated clinical triage system.

---

## 5. STT / TTS engines

**Default, production path (cascade):**
- STT: shared `faster-whisper distil-large-v3` via `stt_service/server.py`
  (engine `whisper_shared`). Falls back automatically to the in-process
  `plugins/whisper_stt.py` (engine `whisper`) if the shared service is
  unreachable at worker startup.
- TTS: Qwen3-TTS via vLLM-Omni on a second machine ("PC2", engine
  `qwen_omni`). If PC2 is unreachable at a worker's startup, falls back
  in order: `qwen_shared` (the fast local option below, no cold start),
  then `qwen_local_subprocess` (a fresh per-worker subprocess, pays a real
  cold-start cost). Both pipelines share this exact fallback chain.

**Fast local alternative — shared Qwen3-TTS 0.6B (engine `qwen_shared`)**:
`tts_service/server.py` loads the model ONCE, GPU-warm, shared by every
room over HTTP — the same "one persistent server" pattern
`stt_service/server.py` uses for STT. Start it with:
```bash
python tts_service/server.py   # needs the .venv-voice stack - see tts_service/requirements.txt
```
then set `system_config.json`'s `tts.engine` to `"qwen_shared"`.

**The real tradeoff**: this engine uses *this* machine's GPU budget
(competing with vLLM + shared STT for VRAM) instead of PC2's. Choose
based on your deployment: one box with no PC2 → `qwen_shared`; multiple
concurrent callers and PC1 GPU headroom to spare → the default
`qwen_omni`. `tts_service/server.py` serializes requests through a single
lock (concurrent generation was tested and found to corrupt output).

**Candidate engines** (real, working, each in their own isolated
virtualenv on the host, **not** included in the Docker images):

| Engine | Type | Needs | License note |
|---|---|---|---|
| Parakeet TDT (STT) | subprocess | `.venv-parakeet`, own NeMo build | CC-BY-4.0 |
| Canary 180M Flash (STT) | subprocess | same NeMo venv as Parakeet | CC-BY-4.0 (use *flash*, not `canary-1b`) |
| Chatterbox (TTS) | subprocess | `.venv-chatterbox`, pins `torch==2.6.0` | see resemble-ai/chatterbox |
| Kokoro-82M (TTS) | subprocess | `.venv-kokoro` | Apache-2.0 |
| Piper (TTS) | subprocess, CPU-only | `.venv-piper` | **GPL-3.0-or-later** — get legal sign-off before commercial use |

Each is selected via `system_config.json`'s `stt`/`tts.engine`; check
`models_registry.json` for which entries are **verified** vs. **candidate**
before trusting one in production.

---

## 6. The direct-audio pipeline

`direct_audio_agent/` is a second, independent voice pipeline: caller
audio goes **straight into Gemma 4 12B Unified** — the same LLM the
cascade already uses for text — instead of through a separate
speech-to-text model first. It reuses `RiversideReceptionist` (persona,
5 hospital tools, safety gate) directly from `app/main.py` by import, not
by copy, so it can never silently drift from the cascade's hard-won
behavior. Nothing in `app/`, `models/`, or the shared vLLM/docker config
was changed to build it, beyond what's noted below.

### Why it exists

Gemma 4 12B Unified is **encoder-free** — it has no separate audio tower,
and projects raw audio waveforms directly into the LLM's own embedding
space (confirmed in the model's own README and `config.json`). That
capability was already flagged in this project's vLLM launch flags
(`--limit-mm-per-prompt '{"audio": 1}'`) but had never been exercised. The
real question worth answering: does skipping a dedicated STT model
actually make a voice agent faster, end to end?

### Architecture

| File | Role |
|---|---|
| `gemma_audio_client.py` | Low-level HTTP client: raw audio → `input_audio` chat-completion content part → the shared vLLM server. `transcribe_audio()` (cheap, transcript-only) and `respond_to_audio()` (one call, audio → final reply). |
| `stt_plugin.py` | `GemmaDirectAudioSTT` — a real `stt.STT` plugin wrapping the cheap transcribe call, so it slots into LiveKit's existing turn-detection/safety-gate machinery unchanged. |
| `llm_plugin.py` | `GemmaDirectAudioLLM` — a real `llm.LLM` plugin generating the actual reply (including tool calls) from the SAME raw audio in one streamed call — this is the design that beats the cascade (see below). |
| `agent.py` | The LiveKit worker entrypoint. Imports `RiversideReceptionist` straight from `app/main.py`, unmodified. Uses explicit dispatch (`agent_name="direct-audio-receptionist"`). |
| `call_server.py` | Lets you call this agent from a browser — mints tokens with explicit dispatch, separate port (`7862`) from `app/token_server.py`. |
| `benchmark.py` | Real, measured latency/accuracy comparison across the cascade and two direct-audio variants. |
| `tests/` | 47 tests (unit + live) covering both new plugins and the client. |

### Why one combined call, not transcribe-then-generate

The first working version of this pipeline swapped Whisper for
Gemma-as-transcriber but kept a *separate* text-only LLM call for the
real reply — safe, but measurably **slower** than the cascade (a real,
reported regression, not a win). The design that actually shipped
collapses this to one audio-conditioned Gemma call for the real reply,
with a second, cheap, transcript-only Gemma call running in parallel
purely to feed LiveKit's turn-detection/safety-gate machinery (which
needs real transcript text, but — verified directly in LiveKit's source —
does *not* need it to come from the same call that produces the reply).

Real, measured latency comparison (`direct_audio_agent/benchmark.py`,
5 runs/utterance, current server config):

| Utterance | Cascade (text) | Gemma-transcribe + text reply | **Gemma single audio→reply call** |
|---|---|---|---|
| short_command | 485.0ms | 465.2ms | **316.6ms** |
| long_sentence | 556.9ms | 641.3ms | **375.7ms** |
| name_heavy | 438.1ms | 631.2ms | **222.4ms** |
| general_question | 531.2ms | 640.2ms | **384.8ms** |
| **Average** | **502.8ms** | **594.5ms** | **~324.9ms** |

The single-call design is **~35% faster** than the cascade on average.
Isolating just Gemma's own time-to-first-token (excluding network/tool
overhead), text vs. audio input make essentially no difference
(~45ms vs. ~49ms) — the win comes entirely from skipping a second full
round trip, not from the model itself running faster on audio.

### Running it

**Docker** (its own Compose profile, `direct-audio`, separate from the
cascade's `cascade` profile — see [§10](#10-capacity--performance) for
why they're kept separate by default):
```bash
docker compose -f docker/docker-compose.yml --profile direct-audio up -d --build
```
Then open `http://<host>:7862/` — same frontend as `:7860`, a different
worker underneath.

**Bare-metal**, same venv `app/main.py` already uses:
```bash
venv/bin/python direct_audio_agent/agent.py start       # the worker
venv/bin/python direct_audio_agent/call_server.py        # :7862
venv/bin/python -m pytest direct_audio_agent/tests/ -v   # 47 tests
venv/bin/python direct_audio_agent/benchmark.py          # real latency numbers
```

### Current status — honest limitations

This is genuinely experimental, not a drop-in replacement for the
cascade:

- **STT transcript quality on real microphone audio needs more work.**
  Clean synthetic (piper) test audio transcribes perfectly; a real live
  call produced some nonsense transcripts from `GemmaDirectAudioSTT`. The
  reply path (audio-native, not transcript-dependent) proved resilient to
  this in the one real call that hit it, but the transcript is still
  what the safety gate's regex match runs on.
- **No real emergency-phrase test through this pipeline specifically.**
  The deterministic safety gate is reused unmodified, but nobody has
  said something like "I'm having chest pain" on an actual call through
  `GemmaDirectAudioSTT` to confirm the match still fires correctly
  against Gemma's transcript wording.
- **No concurrent-load test.** Every latency number above is a single
  serial request on an otherwise idle GPU.
- **A real booking has never been carried through to completion** on
  this pipeline — the furthest a real call has gone is
  `check_availability` returning real slots.
- **The vLLM audio-dependency fix (`librosa`/`soundfile`/`av`) is not yet
  permanent** — it lives in the running container's writable layer, not
  the image, so any future `docker compose up --build`/`--force-recreate`
  of `vllm` silently loses it again. A small custom Dockerfile layer
  would fix this permanently; not yet done.

---

## 7. Managing doctors & schedules

`app/admin_ui.html` (served at `:7870`, password-protected) has two tabs:

- **Hospital Data** — doctor bios, departments, hours, and other
  informational content the AI quotes to callers (`hospital_data.json`).
- **Doctor Schedules** — the actual bookable calendar
  (`hospital_core/booking.py`'s SQLite `slots` table):
  - **Add a new doctor**: name, department, optional bio, and optionally
    an initial recurring weekly schedule in the same form.
  - **Add timings for an existing doctor**: one-off date/time slots, or a
    recurring weekly pattern.
  - **Remove a timing**: only for slots nobody has booked yet.

Underlying API (`app/admin_server.py`, all behind admin Basic-Auth):
`GET /api/doctors`, `GET /api/doctors/slots`, `POST /api/doctors`,
`POST /api/doctors/slots`, `DELETE /api/doctors/slots`.

**Cross-process note**: agent worker(s) run as separate processes with
their own in-memory copy of hospital data. Admin saves reload/re-embed in
the admin server's own process immediately; agent workers pick up the
change on their *next new call*, not mid-call.

---

## 8. Deployment

Requires the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on both machines (`docker info | grep -i runtime` should list `nvidia`).

**0. Generate real LiveKit credentials (once):**
```bash
docker run --rm livekit/livekit-server generate-keys
# copy the printed key/secret into LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
# and LIVEKIT_KEYS in app/.env - see §4.
```

**PC1** (LLM + one voice pipeline of your choice + the three web
interfaces) — pick a **profile**:
```bash
cp app/.env.example app/.env   # edit for your machine + step 0's keys

# Whisper + LLM cascade (production pipeline) - :7860
docker compose -f docker/docker-compose.yml --profile cascade up -d --build

# OR the experimental direct-audio pipeline (no Whisper at all) - :7862
docker compose -f docker/docker-compose.yml --profile direct-audio up -d --build

# Both at once, if your GPU has the headroom:
docker compose -f docker/docker-compose.yml --profile cascade --profile direct-audio up -d --build
```
Always-on regardless of profile: `livekit-server` (:7880), `vllm`
(:8000, Gemma 4 12B), `admin-server` (:7870), `dev-server` (:7871).
`hospital_core/` is bind-mounted (not baked into the image) so whichever
worker(s) you run and `admin-server` share one real, persistent database.

**Why profiles, not "everything, always"**: measured directly on a single
24GB card — running both pipelines' GPU worker processes simultaneously
caused real contention (Silero VAD fell behind realtime, GPU memory at
86% used). Stopping one pipeline's worker dropped GPU memory by ~3.3GB
immediately. Pick the profile matching whichever pipeline you're actually
testing; pass both `--profile` flags together if your hardware genuinely
has the headroom.

**Managing the stack:**
```bash
docker compose -f docker/docker-compose.yml ps                        # what's running
docker compose -f docker/docker-compose.yml down                      # stop + remove everything
docker compose -f docker/docker-compose.yml stop agent stt-service token-server           # cascade only
docker compose -f docker/docker-compose.yml stop direct-audio-agent direct-audio-call-server  # direct-audio only
```

**PC2** (Qwen3-TTS box) — copy `remote/qwen_tts_server/` and
`models/qwen3-tts/` to the second machine, then:
```bash
docker compose -f docker/docker-compose.pc2.yml up -d --build
```
Point PC1's `QWEN_OMNI_BASE_URL` at this machine's LAN IP, port 8091.

**Optional: fast local TTS (`tts-service`)** — opt-in, not started by a
plain `docker compose up`:
```bash
docker compose -f docker/docker-compose.yml --profile local-tts up -d --build tts-service
```
Then set `system_config.json`'s `tts.engine` to `"qwen_shared"`.

**Note on rebuild speed**: `docker compose up --build` only re-downloads
dependencies when `app/requirements.txt` itself changes — editing
`main.py` and rebuilding is fast (seconds, cached layer).

**Known Docker-specific limitation**: candidate-engine fallbacks that
spawn a subprocess in a separate host venv aren't available inside the
container. If PC2 is unreachable, the containerized `agent` logs the
failure rather than silently falling back to a local subprocess; run
bare-metal instead if you need that fallback.

### Bare-metal (no Docker)

**PC1:**
```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r app/requirements.txt
cd app && python main.py download-files
```
`livekit.yaml` must bind `0.0.0.0`, not just loopback. Then, one terminal
each:
```bash
LIVEKIT_KEYS="<key>: <secret>" livekit-server --config app/livekit.yaml --dev
bash scripts/run_vllm.sh                              # LLM (shared by both pipelines)
python stt_service/server.py                          # shared STT (cascade only)
cd app && python main.py dev                           # cascade agent
uvicorn token_server:app --host 0.0.0.0 --port 7860
uvicorn admin_server:app --host 0.0.0.0 --port 7870
uvicorn dev_server:app   --host 0.0.0.0 --port 7871
```
Direct-audio pipeline, same venv, run alongside the above:
```bash
venv/bin/python direct_audio_agent/agent.py start
venv/bin/python direct_audio_agent/call_server.py       # :7862
```

**PC2:**
```bash
pip install git+https://github.com/vllm-project/vllm-omni.git
vllm serve /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8091
```

---

## 9. Testing

```bash
pip install pytest
pytest tests/test_booking.py -v                        # 3 tests, real booking engine
pytest direct_audio_agent/tests/ -v                     # 47 tests, direct-audio pipeline
```

`tests/test_booking.py` covers: double-booking rejection (sequential and
concurrent), fuzzy department/doctor-name matching, cancel/reschedule
flows, the past-time slot filter, and schedule-management functions.

`direct_audio_agent/tests/` covers: HTTP client encoding/error paths and
the empty-content retry (20 tests), STT plugin buffer/resample plumbing
(5 tests), and LLM plugin streaming/tool-call/thinking-leak-filter logic
plus live tool-calling integration (22 tests).

`tests/manual/` holds ad-hoc scripts that need a GPU and locally-running
services, run by hand rather than in CI:
- `benchmark_latency.py` — real per-component timing (VAD, turn detector,
  STT, LLM, TTS) against whichever services are actually running.
- `e2e_call_latency_test.py` — joins the live room as a synthetic caller
  and measures a full real call end to end.
- Load tests, manual voice/text smoke tests for individual engines.

`direct_audio_agent/benchmark.py` — real, measured 3-way latency/accuracy
comparison between the cascade and both direct-audio variants (§6, §10).

---

## 10. Capacity & performance

Measured on a single RTX 3090 (24GB), current config:
- vLLM (Gemma 4 12B, bfloat16 KV cache — this GPU has no native FP8
  hardware): `--max-num-seqs 4`, `--max-model-len 16384`,
  `--gpu-memory-utilization 0.65`. Verified live: **55,235 tokens** of KV
  cache capacity, **3.37x** worst-case concurrency at a full 16,384-token
  conversation. Real conversations run ~1800-2900 tokens even for a long
  multi-turn call, so 3+ realistic concurrent callers fit comfortably
  within budget. `max-model-len` was raised in two stages (4096 → 8192 →
  16384) as real usage and chat-history bounding made each ceiling safe
  to raise — see `scripts/run_vllm.sh`'s comments for the full,
  numbered derivation.
- Marginal cost per concurrent caller (STT + local TTS fallback, not
  vLLM): ~4.6GB with Qwen-local, ~2.7GB with Kokoro.
- Running both pipelines' GPU workers at once measured at 86% GPU memory
  used with real VAD contention (see §8) — the reason they default to
  separate Compose profiles.
- Offloading TTS to PC2 (or the local `qwen_shared` service, §5) frees
  additional PC1 budget for concurrent STT+LLM callers.

**Real per-turn latency, cascade** (`app/main.py`): once the agent is
confident a caller has finished speaking, it typically starts responding
in well under a second — instrumented live via `latency_log.py`/the
`:7871` dev console for every call.

**Real per-turn latency, direct-audio** (`direct_audio_agent/`), from a
live call:

| Turn | EOU delay | STT (Gemma) | LLM ttft | TTS ttfb |
|---|---|---|---|---|
| 2 | 0.67s | 0.62s | 0.06s | 0.23s |
| 3 | 0.64s | 0.63s | 0.06s | 0.49s |
| 4 | 0.59s | 0.60s | 0.13s | 0.46s |
| 5 | 0.61s | 0.58s | 0.08s | 0.35s |

See `PROJECT_PRESENTATION.md` §6 for the complete set of measurements
(streaming head-start numbers, before/after bug-fix comparisons, GPU
memory deltas) if you need the full data set rather than the summary
above.

---

## 11. Troubleshooting

**Caller device connects but the mic never activates / nothing happens
when you talk**: browsers block microphone access on plain `http://`
unless the origin is `localhost` or explicitly treated as secure. Android
Chrome: `chrome://flags/#unsafely-treat-insecure-origin-as-secure` → add
your `http://<LAN-IP>:7860`. Desktop Chrome supports the same flag.
iPhone Safari needs a local HTTPS cert (mkcert) instead.

**"Could not establish peer connection" / connects on one device but not
another on the same network**: a WebRTC/UDP problem, not an application
bug. This self-hosted setup has no TURN relay, so it needs direct UDP
connectivity between the caller's device and the server on ports
`20000-21000` (+ TCP `7881`). Most common causes, roughly in order:
- A **VPN active on the client device** — disconnect it and retry.
- The client device's own **firewall** silently blocking outbound UDP.
- **WiFi client isolation** on the network — use a non-isolated network.

**Long conversation goes silent partway through**: see
[§4](#4-configuration--reliability) — this was a real bug (unbounded chat
history hitting the LLM's context limit) that's now fixed. If it recurs,
check the agent's logs for an HTTP 400 from vLLM mentioning "maximum
context length."

**A pipeline "isn't working" / a call goes silent or errors**: check the
relevant container's logs first (`docker logs <container> --since 10m`),
and check `nvidia-smi` for GPU memory pressure if both pipelines are
running at once. `PROJECT_PRESENTATION.md` §5 has the complete log of
every real bug found this way, as a reference for what symptoms have
turned out to mean in the past.

---

## 12. Repository layout

```
voice-agent-pipeline/
├── app/                      # the cascade agent + its FastAPI control surfaces
│   ├── main.py                 LiveKit worker entrypoint (agent persona,
│   │                            function tools, safety gate, session wiring)
│   ├── token_server.py         mints join tokens + serves frontend/ (:7860)
│   ├── admin_server.py         hospital data + doctor/schedule admin (:7870)
│   ├── dev_server.py           model/backend switcher, latency compare (:7871)
│   ├── helpers.py              run_with_filler, push_ui, credential fail-closed
│   ├── system_config.py/.json  which LLM/STT/TTS is active right now
│   ├── models_registry.json    curated model/engine catalogue for the dev UI
│   ├── hospital_core/          booking.py (SQLite), hospital_kb.py, rag.py,
│   │                            safety.py - the domain logic
│   ├── plugins/                STT/TTS engine adapters (see §5)
│   ├── frontend/index.html     caller-facing web UI (shared by both pipelines)
│   ├── admin_ui.html            hospital data + doctor schedule admin UI
│   ├── requirements.txt
│   └── .env                    local config (see §4) - gitignored
├── direct_audio_agent/        experimental: raw audio straight into Gemma,
│                                no separate STT model at all (see §6)
├── stt_service/               shared faster-whisper STT server (see §2)
├── tts_service/                shared local Qwen3-TTS 0.6B server (see §5)
├── remote/qwen_tts_server/    PC2 (TTS box) Dockerfile + load test
├── docker/                    all Dockerfiles + compose files (see §8)
├── scripts/run_vllm.sh        vLLM launch script (bare-metal, non-Docker)
├── chat_templates/            Gemma tool-calling chat template
├── tests/
│   ├── test_booking.py         real pytest coverage (see §9)
│   └── manual/                 ad-hoc load/latency/smoke scripts, not CI
├── PROJECT_PRESENTATION.md    full build history, every bug/measurement, in detail
└── models/, archive/, remote_services/, data/   gitignored - see below
```

**Deliberately gitignored / not committed:**
- `models/` — model weights, fetched separately.
- `archive/` — a dead-end STT experiment (a standalone Nemotron ASR
  server) kept on disk for reference, superseded by `stt_service/`.
- `remote_services/` — a local vendored clone of `vllm-project/vllm-omni`
  used during development; `remote/qwen_tts_server/` is the real,
  tracked, minimal PC2 setup.
- `data/`, stray `*.wav`/`*.db` files — scratch/runtime output.
- The per-engine virtualenvs at the repo root (`.venv-parakeet`,
  `.venv-kokoro`, etc.) — see [§5](#5-stt--tts-engines) for why they exist.
