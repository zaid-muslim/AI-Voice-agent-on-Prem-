# Voice Agent Pipeline

Local, real-time voice assistant: speech in → transcription → LLM response → cloned-voice speech out.
Currently configured as a bank branch receptionist: it answers informational questions
(branches, hours, services, contact info, and detailed product/fee/rate/policy questions grounded
in its own documents) and can take two actions — block a lost/stolen card after identity
verification, and log a callback request from a human representative. It still cannot check
balances or make transactions.

## Domain config

All brand/company-specific data lives in `config/bank_config.json` — bank name, branches, hours,
services, and contact info. To retarget the bot at a different bank or company, just edit
that file (it's re-read on every turn, no restart needed) and update the persona wording in
`build_system_prompt()` in `src/server.py` if the domain changes (e.g. from a bank to a clinic).
The bot greets automatically the moment a call connects (a WebSocket connection opens) —
before the caller says anything — using the same LLM+TTS pipeline as any other turn, so it
can be interrupted like any other response.

## Architecture

```
Speech Input → Speech-to-Text (STT) → Response Generation (LLM) → Text-to-Speech (TTS) → Voice Output
```

`src/server.py` (STT + LLM orchestration) and `src/chatterbox_server.py` (TTS microservice) run as separate
processes in separate Python environments.

## Repository layout

```
src/       Python services — server.py (STT+LLM+WebSocket), chatterbox_server.py (TTS),
           gen_reference.py, banking.py (verification/handoff logic), db.py (SQLite), seed_db.py,
           rag.py (always-on retrieval), build_index.py (offline doc ingestion), convo_log.py
web/       Browser frontend — index.html, pcm-worklet.js (served statically)
config/    bank_config.json — swappable domain/brand config
           rag_docs/ — Markdown source docs for always-on RAG (checked in, human-edited)
data/      memory.json (long-term memory), bank.db (SQLite: customer/audit/handoff/rag_chunks),
           rag_index.npy (embedding matrix) — all gitignored, regenerable
assets/    voice_seed/ (TTS reference clips), audio_samples/ (sample recordings)
tests/     pytest suite for card-block/handoff logic (test_banking.py) and RAG retrieval/chunking
           logic (test_rag.py)
Plans/     Design/planning docs
logs/      Runtime logs (gitignored)
run.sh     Launches the whole pipeline
```

## Models

| Stage | Model | Parameters | Precision / Quantization | Runs on |
|-------|-------|------------|---------------------------|---------|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3` | ~1.55B | INT8 (`int8_float16` compute type) | GPU |
| LLM | Qwen2.5 14B Instruct (via vLLM) | 14.8B | AWQ (4-bit) | GPU |
| TTS | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) | ~0.5B backbone (T3) + S3Gen vocoder | Voice-cloned from a reference clip | GPU |

## Features

- **Real-time listening, no push-to-talk** — click the mic once to toggle continuous
  listening on. The browser streams raw 16kHz PCM audio to the server via an
  `AudioWorklet` (`pcm-worklet.js`); the server runs [`webrtcvad`](https://github.com/wiseman/py-webrtcvad)
  frame-by-frame to detect when you start and stop talking, and only then transcribes
  and responds — no manual recording step. (`pcm-worklet.js` lives in `web/`.)
- **Automatic barge-in** — the moment the server's VAD confirms you've started talking
  again (even mid-response), it immediately cancels whatever the agent is doing (LLM +
  TTS) and starts processing your new utterance. No button press needed.
- **Conversation memory** — history persists per browser session (last ~10 exchanges) via
  vLLM's OpenAI-compatible `/v1/chat/completions` endpoint.
- **Long-term memory** — durable facts about the caller (name, or anything explicitly asked
  to be remembered) survive across sessions via a `remember` tool, stored in `data/memory.json`
  and injected into the system prompt every turn. Delete `data/memory.json` to wipe it.
- **Web search** — scoped strictly to bank/finance-relevant lookups (e.g. exchange rates)
  backed by a local [SearXNG](https://github.com/searxng/searxng) instance
  (`http://localhost:1234`). Anything unrelated to banking gets redirected, not searched.
- **Scope guarding** — the bot stays in character as the bank's receptionist: it won't
  fabricate a branch, phone number, or policy that isn't in `config/bank_config.json`, won't attempt
  account lookups or transactions (points callers to the app, a branch, or customer care
  instead), and redirects off-topic questions back to banking.
- **Card blocking with identity verification** — if a caller wants to block a lost/stolen card,
  the agent collects three factors one at a time (card last 4, mother's maiden name, date of
  birth) and calls a `block_card` tool. **All matching happens in Python** (`src/banking.py`),
  never in the LLM — the model only collects answers and never learns whether they were right.
  The tool result is a bare status code (`blocked` / `declined` / `handed_off`), so no
  prompt-injection can extract the stored answers, and unknown cards are reported identically to
  wrong answers (no card enumeration). Two failed attempts per call auto-queues a human handoff.
  Every attempt is written to an `audit_log` table. Customer data lives in `data/bank.db` (SQLite).
  A server-side **guardrail** is the last line of defence: the agent is physically prevented from
  voicing (or recording) a "card blocked / identity verified" confirmation unless `block_card`
  actually returned `blocked` this turn — any hallucinated confirmation is suppressed from the
  audio and replaced with a truthful, server-authored line.
- **Human handoff (callback ticket)** — there's no telephony/transfer layer yet, so instead of
  giving out the bank's number (which loops back to this very agent), a request for a human calls
  `request_human_handoff`, which queues a ticket in `data/bank.db` and promises a callback within
  one business day.
- **Grounded business-info answers (always-on RAG)** — detailed product/fee/rate/policy questions
  are answered from the document corpus, not the LLM's own knowledge. Retrieval is **not** a tool
  the model chooses to call: the server retrieves on **every** turn (`src/rag.py`) and injects the
  matching chunks into context before the LLM runs — so grounding never depends on the model
  deciding to look something up, and it answers in a single LLM pass. The retrieval query is
  context-aware (raw utterance + last exchange, hybrid-merged) so pronoun follow-ups ("what's the
  eligibility for *it*?") resolve to the right document. Runs entirely on CPU (fastembed,
  `BAAI/bge-small-en-v1.5`, ONNX INT8) — no GPU/VRAM contention with Whisper/vLLM/Chatterbox —
  over a flat NumPy cosine-similarity index built from Markdown docs in `config/rag_docs/` by
  `src/build_index.py`. When nothing clears the similarity threshold, no context is injected and the
  prompt directs the agent to say it doesn't have that on file rather than guess. Document/section
  names are never read aloud. (Swapping the NumPy index for a vector DB is a localized change in
  `rag.py` when the corpus outgrows in-memory search.)
- **Conversation + latency log** — every turn is appended to `logs/conversation.log` (gitignored):
  the caller's input, the exact retrieved chunks with similarity scores, any tools called, the
  spoken reply, and per-turn latency (`ttft` = query → first spoken word, retrieval time, total).
  Built for diagnosis — a factual reply with weak/empty retrieval is a fabrication, visible at a glance.
- **File upload fallback** — the 📁 button lets you upload an audio file directly
  (any format ffmpeg can decode) instead of using the mic.

## Usage

**Prerequisites:** the SearXNG docker container up (`docker start <container>`, listening on host
port 1234). vLLM itself is started by `run.sh` — no separate service to launch by hand.

**One-time setup:** seed the customer database used for card-block verification:
```bash
python3 src/seed_db.py
```
This creates `data/bank.db` with a few fake test customers (re-run any time to reset them).

Also build the business-doc search index (needed for the always-on RAG retrieval to return real
results — otherwise it finds nothing and the agent says it has no info on file):
```bash
python3 src/build_index.py
```
This chunks the Markdown docs in `config/rag_docs/`, embeds them, and writes `data/bank.db`'s
`rag_chunks` table plus `data/rag_index.npy`. Safe to re-run any time the docs change.

Run the tests with `pytest tests/ -q`.

**Quick start:**
```bash
./run.sh
```
This starts Chatterbox, the file server, and the main server together, and stops everything
cleanly on `Ctrl+C`.

**Manual steps** (equivalent to the above):
1. Start vLLM: `vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --served-model-name qwen2.5-14b-awq --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.5 --max-model-len 8192 --port 8000`
2. Start the Chatterbox TTS service: `.chatterbox-venv/bin/python3 src/chatterbox_server.py`
3. Start the main server: `python3 src/server.py`
4. Serve the UI: `python3 -m http.server 3000 --directory web`
5. Open the app at [http://127.0.0.1:3000](http://127.0.0.1:3000)

