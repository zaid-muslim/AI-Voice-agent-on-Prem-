# Voice Agent Pipeline

Local, real-time voice assistant: speech in → transcription → LLM response → cloned-voice speech out.
Currently configured as a bank branch receptionist — conversational and informational only
(branches, hours, services, contact info); no account access or transactions yet.

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
src/       Python services — server.py (STT+LLM+WebSocket), chatterbox_server.py (TTS), gen_reference.py
web/       Browser frontend — index.html, pcm-worklet.js (served statically)
config/    bank_config.json — swappable domain/brand config
data/      memory.json — runtime long-term memory (gitignored)
assets/    voice_seed/ (TTS reference clips), audio_samples/ (sample recordings)
Plans/     Design/planning docs
logs/      Runtime logs (gitignored)
run.sh     Launches the whole pipeline
```

## Models

| Stage | Model | Parameters | Precision / Quantization | Runs on |
|-------|-------|------------|---------------------------|---------|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3` | ~1.55B | INT8 (`int8_float16` compute type) | GPU |
| LLM | Qwen2.5 14B (via Ollama) | 14.8B | Q4_K_M (4-bit) | GPU |
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
  Ollama's `/api/chat`.
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
- **File upload fallback** — the 📁 button lets you upload an audio file directly
  (any format ffmpeg can decode) instead of using the mic.

## Usage

**Prerequisites:** Ollama running, and the SearXNG docker container up (`docker start <container>`,
listening on host port 1234).

**Quick start:**
```bash
./run.sh
```
This starts Chatterbox, the file server, and the main server together, and stops everything
cleanly on `Ctrl+C`.

**Manual steps** (equivalent to the above):
1. Start Ollama: `ollama serve`
2. Start the Chatterbox TTS service: `.chatterbox-venv/bin/python3 src/chatterbox_server.py`
3. Start the main server: `python3 src/server.py`
4. Serve the UI: `python3 -m http.server 3000 --directory web`
5. Open the app at [http://127.0.0.1:3000](http://127.0.0.1:3000)

