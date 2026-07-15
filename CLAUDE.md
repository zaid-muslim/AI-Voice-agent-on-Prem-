# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, real-time voice assistant: speech in → transcription → LLM response → cloned-voice speech
out. Currently configured as a bank branch receptionist (HBL) that answers informational questions
grounded in its own documents, and can take exactly two actions: block a lost/stolen card after
identity verification, and log a human-callback request. It cannot check balances or make
transactions.

## Running it

```bash
./run.sh
```

Starts vLLM, the Chatterbox TTS microservice, a static file server for `web/`, and the main
server together, in that dependency order, and stops everything cleanly on Ctrl+C. Building the
RAG index (see below) happens automatically on first run if `data/rag_index.npy` is missing.

**Prerequisite:** a SearXNG docker container must already be running on host port 1234
(`docker start <container>`) for web search to work — `run.sh` warns but doesn't fail if it's down.

**Manual/equivalent steps**, useful when iterating on one piece:
1. `vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --served-model-name qwen2.5-14b-awq --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.5 --max-model-len 8192 --port 8000` (or `./_launch_vllm.sh`, which sets CUDA env vars for this box)
2. `.chatterbox-venv/bin/python3 src/chatterbox_server.py` — TTS microservice, separate venv (needs its own torch/chatterbox install)
3. `python3 src/server.py` — main STT+LLM+WebSocket server (miniconda python, not the chatterbox venv)
4. `python3 -m http.server 3000 --directory web`
5. Open http://127.0.0.1:3000

**One-time / as-needed setup:**
```bash
python3 src/seed_db.py       # (re)creates data/bank.db with fake test customers for card-block verification
python3 src/build_index.py   # chunks config/rag_docs/*.md, embeds them, writes data/bank.db's rag_chunks + data/rag_index.npy
```
Re-run `build_index.py` any time the docs in `config/rag_docs/` change — it always rebuilds from
scratch. Without a built index, every product/fee/policy question gets "I don't have that on file."

## Tests

```bash
pytest tests/ -q
```
`test_banking.py` covers card-block/handoff logic in `src/banking.py` against an in-memory SQLite
DB — no LLM involved, since that logic is deliberately deterministic Python. `test_rag.py` covers
chunking and retrieval logic in `src/rag.py` / `src/build_index.py`.

## Architecture

```
Speech Input → STT (faster-whisper) → LLM (vLLM, Qwen2.5-14B-AWQ, tool calling) → TTS (Chatterbox) → Voice Output
```

`src/server.py` (STT + LLM orchestration + WebSocket) and `src/chatterbox_server.py` (TTS) run as
**separate processes in separate Python environments** — Chatterbox needs its own venv
(`.chatterbox-venv/`) for its torch/chatterbox dependency stack, independent of the main server's
(miniconda) environment. They talk over HTTP (`localhost:8766/synthesize`). vLLM is a third
process, spoken to over its OpenAI-compatible `/v1/chat/completions` endpoint.

All three GPU consumers (Whisper STT, vLLM, Chatterbox TTS) share one GPU — memory budgeting is
deliberate throughout (`gpu_memory_utilization=0.5-0.62`, `int8`/AWQ quantization) and any change
to one service's footprint can starve the others.

### Turn lifecycle (`src/server.py`)

Every caller turn (mic audio, uploaded file, or typed text) funnels into
`respond_to_transcript()`, which:
1. Runs always-on RAG retrieval (`rag.search_docs_multi`) **before** the LLM call — retrieval is
   never a tool the model chooses to call, it's injected into context as an ephemeral system
   message every turn (see "Always-on RAG" below).
2. Streams the LLM response (`stream_llm`) sentence-by-sentence into TTS (`speak_and_return` /
   `speak_stream`) so audio starts before the full reply is generated.
3. Dispatches any tool calls the model made (`web_search`, `remember`, `block_card`,
   `request_human_handoff`) and, for `block_card`, speaks the outcome **deterministically from the
   tool result**, never from what the model wrote (see guardrail below).
4. Commits only what actually happened to `history` in a `finally` block — so a barge-in
   mid-turn still leaves history consistent — and best-effort logs the turn to
   `logs/conversation.log` via `convo_log.py`.

Everything is a cancellable `asyncio.Task` per connection (`state["current_task"]` in
`handle_client`); barge-in works by cancelling the in-flight task and starting a new one, not by
some separate interrupt flag threaded through the LLM/TTS calls.

### Real-time VAD / barge-in

The browser streams raw 16kHz PCM continuously over the WebSocket via an `AudioWorklet`
(`web/pcm-worklet.js`); `src/server.py` runs `webrtcvad` frame-by-frame plus an RMS energy gate
(`SPEECH_RMS_THRESHOLD`) to reject the agent's own echo/background noise. Barge-in requires more
sustained speech (`BARGE_IN_FRAMES`) than starting fresh from idle (`SPEECH_START_FRAMES`) so a
brief blip doesn't cut off the agent. All the VAD tuning constants live at the top of
`src/server.py` with comments on what raising/lowering each one does.

### Identity verification is never the LLM's job (`src/banking.py`)

The model only *collects* the three verification factors (card last-4, mother's maiden name, DOB)
in conversation, one at a time — **all matching happens in Python**. The tool result handed back
to the model is a bare status code (`blocked` / `declined` / `handed_off`), never the stored
secrets, so prompt injection can't extract them, and an unknown card is reported identically to a
wrong answer (no card enumeration via response difference). Two failed attempts per call
auto-queues a human handoff (`MAX_CARD_ATTEMPTS` in `banking.py`). Every attempt is written to
`audit_log` in `data/bank.db`.

A **server-side output guardrail** (`banking.asserts_block_success`, applied in
`speak_stream`'s `emit()`) is the last line of defense: any sentence claiming a card was blocked
or an identity verified is suppressed from audio/history unless `block_card` actually returned
`blocked` *this turn* — a hallucinated confirmation gets replaced with a truthful,
server-authored line (`CARD_OUTCOME_LINES` / `UNVERIFIED_BLOCK_FALLBACK`) instead. When touching
this path, preserve the split: collection/wording = LLM, correctness decision = Python, spoken
confirmation of a completed action = server-authored, never model-authored.

### Always-on RAG (`src/rag.py`, `src/build_index.py`)

Business-info retrieval is deliberately **not** a tool call — the server retrieves from the
document corpus on *every* turn and injects matching chunks as an ephemeral system message right
before the caller's question (not persisted to history). This guarantees grounding doesn't depend
on the model deciding to look something up, and keeps replies to a single LLM pass.

- Runs entirely on CPU (`fastembed`, `BAAI/bge-small-en-v1.5`, ONNX INT8) — no GPU/VRAM contention
  with Whisper/vLLM/Chatterbox.
- Retrieval query is **hybrid**: the raw utterance plus a context-expanded query
  (`rag.build_retrieval_query`, folds in recent history) so pronoun follow-ups ("what's the
  eligibility for *it*?") still resolve to the right document. Results from both queries are
  interleaved/deduped (`search_docs_multi`), not score-merged — a confidently-wrong expanded
  query must not bury the raw utterance's correct hit.
- bge models are trained **asymmetrically**: build-time uses `embed()` (passage mode,
  `build_index.py`), query-time uses `query_embed()` (`rag.py`). Mixing these up silently degrades
  results without erroring — watch for this if retrieval quality regresses after a refactor.
- `SIMILARITY_THRESHOLD = 0.60` in `rag.py` was calibrated against this specific corpus + model;
  see the comment there before changing it.
- The flat NumPy matrix (`data/rag_index.npy`) is a regenerable, gitignored artifact whose row
  order must match `rag_chunks` table insertion order (`db.py` asserts this at build time).
  Swapping it for a real vector DB is meant to be a localized change inside `rag.py`.

### Domain config is data, not code

`config/bank_config.json` holds all brand-specific data (bank name, branches, hours, services,
contact info) and is **re-read fresh on every prompt build** (`load_domain_config()` in
`server.py`) — no restart needed to edit it. To retarget the bot at a different bank/company,
edit that file and, if the domain itself changes (e.g. bank → clinic), update the persona wording
in `build_system_prompt()` in `src/server.py`.

### Streaming/tool-call wire format note

`stream_llm()` in `server.py` absorbs a real difference from the Ollama-based version this
replaced: vLLM's OpenAI-compatible streaming fragments each tool call across many chunks
(partial JSON string pieces keyed by index) rather than sending one complete parsed tool call.
`stream_llm` reassembles these internally so everything downstream still sees one
`("tool_calls", [...])` event in the same shape as before.

## Repository layout

```
src/       server.py (STT+LLM+WebSocket, the orchestrator), chatterbox_server.py (TTS, separate venv),
           banking.py (verification/handoff, no LLM), db.py (SQLite access), seed_db.py,
           rag.py (always-on retrieval), build_index.py (offline doc ingestion), convo_log.py,
           gen_reference.py, show_db.py
web/       Browser frontend served statically — index.html, pcm-worklet.js (AudioWorklet for mic capture)
config/    bank_config.json (swappable domain/brand config), rag_docs/*.md (human-edited RAG source docs)
data/      memory.json, bank.db (SQLite), rag_index.npy — all gitignored, all regenerable
assets/    voice_seed/ (TTS reference clips for voice cloning), audio_samples/
tests/     pytest suite — test_banking.py, test_rag.py
Plans/     design/planning docs
logs/      runtime logs (gitignored) — conversation.log is the per-turn diagnostic log
```

## Diagnosing issues

`logs/conversation.log` records, per turn: the caller's input, the exact retrieved RAG chunks with
similarity scores, any tools called, the spoken reply, and per-turn latency
(`ttft` = query → first spoken word, retrieval time, total). A factual-sounding reply next to
weak/empty retrieval in this log is a fabrication — that's the intended signal, check here first
when the agent seems to be making things up or ignoring documents.
