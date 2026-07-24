# Riverside General — SOTA Voice Receptionist — Technical README

A self-hosted AI phone receptionist for a hospital, built on LiveKit
Agents, spanning two machines: **PC1** (the main agent + LLM + STT) and
**PC2** (a dedicated TTS box running Qwen3-TTS via vLLM-Omni).

---

## 1. Architecture overview

```
Caller's browser/phone
        │
        ▼
livekit-server (PC1, signaling + media, self-hosted)
        │
        ▼
agent.py (PC1) ── AgentSession(vad, stt, llm, tts, turn_detection)
        │                │            │            │
        │                │            │            └─ HTTP → PC2's
        │                │            │                vLLM-Omni server
        │                │            │                (Qwen3-TTS)
        │                │            └─ HTTP → vLLM (PC1, Gemma 4 12B)
        │                └─ whisper / Parakeet / Canary (local or subprocess)
        ▼
hospital_core/ (booking.py, safety.py, rag.py, hospital_kb.py)
```

**Three web interfaces:**
| Interface | Port | File | Purpose |
|---|---|---|---|
| Caller | 7860 | `token_server.py` | The person phoning in |
| Hospital admin | 7870 | `admin_server.py` | Doctors/departments/hours |
| Developer console | 7871 | `dev_server.py` | Model switching, latency comparison |

**Config layer**: `system_config.json` is the single source of truth for
which LLM/STT/TTS is active. Schema: each section is `{"engine": ...,
"model": ..., "display_name": ...}`. Read fresh by `agent.py` at the
start of every call — a dev-console change takes effect on the next
call, not instantly (LLM switches additionally cost real ~15-60s+
downtime while vLLM restarts; TTS/STT switches are near-instant except
for the cross-process worker-pool caveat below).

---

## 2. The two-machine split

**PC1** runs:
- `livekit-server` (self-hosted signaling/media)
- `agent.py` (the LiveKit worker process)
- vLLM serving the LLM (Gemma 4 12B by default)
- STT (faster-whisper by default; Parakeet/Canary as isolated-venv
  subprocess alternatives)
- `token_server.py`, `admin_server.py`, `dev_server.py`

**PC2** runs:
- vLLM-Omni serving Qwen3-TTS (0.6B or 1.7B CustomVoice), exposing an
  OpenAI-compatible `/v1/audio/speech` endpoint
- One persistent, shared, GPU-warm process — **not** spawned per call.
  Every session on PC1 hits this same server over the network.

**Why this split**: TTS on a dedicated box means PC1's GPU budget is
freed for STT + LLM, and PC2's TTS server stays warm and shared across
every concurrent caller — the same "one persistent server, many
concurrent requests" pattern vLLM already uses for the LLM, now applied
to TTS too.

---

## 3. Environment variables (PC1's `.env`)

```bash
# LiveKit
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_WS_URL=ws://<PC1_LAN_IP>:7880   # MUST be LAN IP for phone/cross-device testing

# LLM (vLLM, local on PC1)
VLLM_BASE_URL=http://127.0.0.1:8000/v1

# Qwen3-TTS via vLLM-Omni on PC2 — REQUIRED, not optional
QWEN_OMNI_BASE_URL=http://<PC2_LAN_IP>:8091/v1
QWEN_OMNI_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice   # or 0.6B
QWEN_OMNI_VOICE=vivian

# STT
WHISPER_MODEL=distil-large-v3
VAD_MIN_SILENCE=0.4
PARAKEET_PYTHON=/path/to/.venv-parakeet/bin/python   # optional, own venv
PARAKEET_WORKER=/path/to/plugins/parakeet_worker.py

# Local-fallback Qwen (legacy path, only used if engine=qwen_local_subprocess)
QWEN_PYTHON=/path/to/.venv-voice/bin/python
QWEN_MODEL_ID=/path/to/Qwentts
QWEN_SPEAKER=aiden

# Admin / dev console auth — CHANGE FROM DEFAULTS BEFORE EXPOSING ON LAN
ADMIN_PASSWORD=change-me
DEV_PASSWORD=change-me
VLLM_PYTHON=/path/to/vllm_venv/bin/python
```

**Real bug this project hit twice**: `QWEN_OMNI_*` variables silently
missing from `.env` meant the agent ran on hardcoded Python fallback
defaults regardless of what was actually configured on PC2. Always
`grep QWEN_OMNI .env` to confirm the lines genuinely exist before
assuming a config change took effect.

---

## 4. Setup — PC1 (main agent)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "livekit-agents[openai,silero,turn-detector]~=1.0" \
            livekit-api livekit fastapi uvicorn python-dotenv loguru \
            faster-whisper aiohttp sentence-transformers
cp .env.example .env   # then edit every path/IP for your machine
python agent.py download-files
```

**LiveKit server config** (`livekit.yaml`) — must bind to all interfaces,
not just loopback, or nothing outside `localhost` can connect:
```yaml
port: 7880
bind_addresses:
  - "0.0.0.0"
rtc:
  tcp_port: 7881
  port_range_start: 50000
  port_range_end: 60000
keys:
  devkey: secret
```
```bash
livekit-server --config livekit.yaml --dev
```

**vLLM (Gemma 4 12B)**:
```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve \
    /path/to/gemma-4-12b-w4a16 \
    --served-model-name gemma-4-12b --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 --max-num-seqs 4 --gpu-memory-utilization 0.40 \
    --kv-cache-dtype bfloat16 --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --chat-template /path/to/tool_chat_template_gemma4.jinja
```
`--kv-cache-dtype bfloat16`, not `fp8` — most consumer GPUs (e.g. RTX
3090, compute capability 8.6) have no native FP8 hardware.

Then, one terminal each: `python agent.py dev`, `uvicorn token_server:app
--host 0.0.0.0 --port 7860`, `uvicorn admin_server:app --host 0.0.0.0
--port 7870`, `uvicorn dev_server:app --host 0.0.0.0 --port 7871`.

---

## 5. Setup — PC2 (Qwen3-TTS box)

```bash
python3 -m venv ~/.venv-qwen-tts
source ~/.venv-qwen-tts/bin/activate
pip install git+https://github.com/vllm-project/vllm-omni.git
```

Download models locally (recommended, avoids re-fetching from HF cache
on every restart):
```bash
pip install -U "huggingface_hub[cli]"
mkdir -p ~/models/qwen3-tts && cd ~/models/qwen3-tts
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Serve (one model at a time, on one GPU):
```bash
vllm serve ~/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8091
```

Verify:
```bash
curl http://127.0.0.1:8091/health
curl -X POST http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "...", "input": "Hello, this is a test.", "voice": "vivian", "response_format": "wav"}' \
  --output test.wav
```

**Real, confirmed bug and fix**: LiveKit's `openai.TTS()` plugin defaults
to a streaming request with an mp3-family format, which vLLM-Omni's
`/v1/audio/speech` rejects outright (`400: Streaming requires
response_format='pcm' or 'wav'`). Fix: pass `response_format="wav"`
explicitly in the `openai.TTS()` constructor on PC1 — confirmed as a
real, present kwarg via `help(openai.TTS.__init__)` on the actual
installed version, not assumed.

**Second real, confirmed bug**: never trust `system_config.json`'s
`tts.model` field for this engine — it can hold a stale value from the
old local-subprocess schema (a speaker name like `"aiden"`, not a valid
model identifier), causing a real `404: The model 'aiden' does not
exist.` Always hardcode `model=QWEN_OMNI_MODEL` in `_make_tts()` for this
engine rather than trusting `tts_cfg["model"]`.

---

## 6. The full model lineup

**LLMs** (via vLLM on PC1): Gemma 4 12B w4a16 (✅ verified), Llama 3.1 8B,
Qwen2.5 7B, Mistral 7B (candidates), custom.

**STT**: whisper distil-large-v3 (✅ verified), whisper large-v3
(candidate, same plugin code), Parakeet TDT (✅ verified, isolated venv +
dedicated-background-event-loop fix), Canary 180m-flash (candidate — use
this variant specifically; `canary-1b` is CC-BY-NC-4.0, non-commercial).

**TTS**: **Qwen3-TTS via vLLM-Omni on PC2** (✅ verified, current default —
0.6B or 1.7B CustomVoice), Qwen local-subprocess (legacy fallback, still
selectable via `engine=qwen_local_subprocess`), Chatterbox (candidate,
non-streaming, own venv), Kokoro (candidate, genuinely streams
per-segment, Apache-2.0), Piper (candidate, CPU-only, **GPL-3.0-or-later
— get legal sign-off before commercial use**).

---

## 7. Real bugs found and fixed (chronological)

1. `initialize_process_timeout` too short for cold loads under GPU
   contention → raised to 300s.
2. Parakeet subprocess event-loop bug: `asyncio.run()` in a sync
   `prewarm()` context created a throwaway loop that died, orphaning the
   subprocess → fixed with a persistent background-loop-in-a-thread
   pattern (later reused correctly for Canary).
3. `_warm_up_vllm()` silently reported "OK" on HTTP errors → fixed to
   check status code.
4. `livekit-server` bound to `127.0.0.1` by default → blocked all
   non-localhost devices. Fixed via `livekit.yaml`'s `bind_addresses`.
5. Stale duplicate `livekit-server` process left running after a config
   change → `address already in use`; required `kill -9` + `ss -tlnp`
   verification before restart.
6. Dead config vars (`GEMMA_MODEL_NAME`, `STT_BACKEND`) left in
   `.env.example` after the schema v2 migration — editing them did
   nothing. Removed.
7. Latency-log sentinel values (`tts_ttfb: -1.0`,
   `end_of_utterance_delay/transcription_delay: 0.0`) recurred for both
   Qwen and Kokoro — corrupting every mean in the dev console's latency
   table. Fixed by discarding any value ≤ 0 before recording.
8. `qwen_worker.py` accidentally deleted, then accidentally placed back
   inside `plugins/` — collided with the bridge file's own name
   (`qwen_tts.py`), causing `from qwen_tts import ...` inside
   `faster_qwen3_tts` to resolve to *our* file instead of the real PyPI
   package, crashing with `ModuleNotFoundError: No module named
   'livekit'`. Fixed by merging bridge + worker into one dual-role file
   (`qwen_tts.py` plays both roles depending on which venv imports it),
   eliminating the naming-collision risk entirely.
9. `LIVEKIT_WS_URL=ws://localhost:7880` — broke cross-device testing,
   since `localhost` on a caller's own device never resolves to PC1.
   Fixed to the real LAN IP.
10. vLLM-Omni `response_format`/`model` bugs — see §5 above.

## Investigated, deliberately NOT changed

- ~3.0s `end_of_utterance_delay` recurring pattern — correlates with a
  3-second AEC-warmup interruption-disable window; flagged as a suspected
  artifact, not filtered, since it isn't provably invalid.
- One genuine 19.35s LLM TTFT outlier — real GPU contention from
  concurrent-caller testing, kept in the data.

---

## 8. Capacity, measured not estimated

- Whisper+Qwen(local): ~4.6GB marginal VRAM/caller.
- Whisper+Kokoro: ~2.7GB marginal VRAM/caller.
- `--max-num-seqs 4` and the GPU memory math both independently land on
  **4 concurrent callers** as the real ceiling on a 24GB card (before the
  PC2 TTS offload — offloading TTS to PC2 frees more of PC1's budget for
  additional concurrent STT+LLM callers, not yet re-measured after the
  split).

---

## 9. Latency breakdown

```
End-of-utterance delay:  ~0.5-0.7s   <- mostly a DELIBERATE safety margin
                                        (VAD_MIN_SILENCE + turn detector),
                                        not raw compute
LLM TTFT:                ~0.1-0.4s
TTS TTFB:                ~0.1-0.3s (PC2, over network)
```

Real current industry replacement for fixed-silence VAD: **semantic
endpointing / End-of-Turn detection** — already partially in use here via
LiveKit's own `EnglishModel` turn detector. **Unresolved lead**:
`endpointing.min_delay` acts as `max(VAD_silence, min_delay)` per
LiveKit's own docs — meaning `VAD_MIN_SILENCE` may be capping the
semantic model's real benefit. Not yet acted on.

---

## 10. Industrial-scale batching (research prototype, in progress)

The LLM already uses the correct pattern: one persistent, shared vLLM
server, batching concurrent requests (`--max-num-seqs`). Now with PC2,
TTS follows the same pattern too (one persistent, shared Qwen-Omni
server). STT still spawns per-session subprocesses.

Built and tested: `inference_batching.py` (a generic `AsyncDynamicBatcher`
implementing NVIDIA Triton's real documented dynamic-batching algorithm —
flush on `max_batch_size` OR `max_queue_delay_ms`, whichever first) and
`parakeet_batch_server.py` (a shared, batched Parakeet STT service using
NeMo's real, verified `model.transcribe(paths, batch_size=N)` API).
**Not yet functionally tested end-to-end** — pending a fake-model
protocol test, same rigor as every other engine in this project.

---

## 11. Working discipline this project has followed

- Every new engine/API gets its real source downloaded and read before
  any plugin code is written — never guessed. Caught real mistakes twice
  (a wrong Piper field-name guess, corrected before shipping; the exact
  vLLM-Omni `response_format` kwarg confirmed via `help()` rather than
  assumed).
- Every new piece of logic gets a real, runnable test against a fake
  stand-in for unavailable hardware — protocol-level subprocess tests,
  timed concurrency tests, not just "looks right."
- Full-project regression (all `.py` files compile, all self-tests, all
  frontend JS) re-run after every change.
- Real license terms checked and flagged plainly (Piper's GPL-3.0,
  Canary's per-variant licensing).
- When a reported "bug" doesn't hold up against real code + real log
  evidence, say so plainly rather than inventing a fix to look responsive.