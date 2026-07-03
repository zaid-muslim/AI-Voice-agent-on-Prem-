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



## Usage

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
