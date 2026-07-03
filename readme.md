# Voice Agent Pipeline

Local, real-time voice assistant: speech in → transcription → LLM response → cloned-voice speech out.

## Architecture

```
Speech Input → Speech-to-Text (STT) → Response Generation (LLM) → Text-to-Speech (TTS) → Voice Output
```

`server.py` (STT + LLM orchestration) and `chatterbox_server.py` (TTS microservice) run as separate
processes in separate Python environments.

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
  and responds — no manual recording step.
- **Automatic barge-in** — the moment the server's VAD confirms you've started talking
  again (even mid-response), it immediately cancels whatever the agent is doing (LLM +
  TTS) and starts processing your new utterance. No button press needed.
- **Conversation memory** — history persists per browser session (last ~10 exchanges) via
  Ollama's `/api/chat`.
- **Web search** — the LLM has a `web_search` tool backed by a local
  [SearXNG](https://github.com/searxng/searxng) instance (`http://localhost:1234`). It's
  instructed to search only when it doesn't already know the answer (news, weather, prices,
  recent events) — not for general knowledge — and to say a short line out loud ("let me
  check that") before searching, so there's no silent gap while it waits on results.
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
2. Start the Chatterbox TTS service: `.chatterbox-venv/bin/python3 chatterbox_server.py`
3. Start the main server: `python3 server.py`
4. Serve the UI: `python3 -m http.server 3000`
5. Open the app at [http://127.0.0.1:3000](http://127.0.0.1:3000)

