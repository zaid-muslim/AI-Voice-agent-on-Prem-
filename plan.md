# Voice Agent Pipeline — Implementation Spec

## Goal
Build a working local voice bot: speak → transcribed → LLM responds → spoken back.  
Deploy to `~/Desktop/Voice agents/Pipeline/` on the 3090 box.

---

## Stack (all confirmed on box)
| Component | Model | How |
|-----------|-------|-----|
| STT | faster-whisper large-v3 INT8 | Python, GPU, HF cache already present |
| LLM | Qwen2.5 14B | Ollama, already pulled (`qwen2.5:14b`) |
| TTS | Orpheus 3B | Python, GPU, downloads on first run |
| Transport | WebSocket port 8765 | `websockets` Python library |

---

## File Structure
```
~/Desktop/Voice agents/Pipeline/
├── server.py        # WebSocket server — STT + LLM + TTS pipeline
├── index.html       # Single-file browser UI
└── requirements.txt
```

---

## requirements.txt
```
faster-whisper>=1.0
orpheus-speech
websockets>=12.0
numpy
soundfile
```

Install with:
```bash
/home/nauyan/miniconda3/bin/python3 -m pip install -r requirements.txt
```

---

## WebSocket Protocol

All messages are JSON. Server sends, client receives:

```jsonc
// Transcript of what the user said
{"type": "transcript", "text": "..."}

// LLM token stream (one per token)
{"type": "token", "text": "..."}

// Audio chunk (one per sentence, base64-encoded WAV, 24kHz mono)
{"type": "audio", "data": "<base64>"}

// Pipeline done for this turn
{"type": "done"}

// Error
{"type": "error", "message": "..."}
```

Client sends to server:
```jsonc
// Raw audio bytes (WebM Opus from MediaRecorder)
<binary frame>
```

---

## server.py — Full Implementation Spec

```python
#!/usr/bin/env python3
"""
Voice agent WebSocket server.
Run: python3 server.py
"""

import asyncio, base64, io, json, re, time, wave
import numpy as np
import soundfile as sf
import websockets
from faster_whisper import WhisperModel
from orpheus_tts import OrpheusTTS    # pip install orpheus-speech

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"
WS_PORT      = 8765
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep responses concise and conversational. "
    "Avoid markdown, bullet points, or code blocks — speak in plain sentences."
)
TTS_VOICE    = "tara"   # Orpheus built-in voices: tara, leah, jess, leo, dan, mia, zac, zoe
SAMPLE_RATE  = 24000    # Orpheus output sample rate

# ── Model Loading (once at startup) ───────────────────────────────────────────
print("Loading STT model...")
stt_model = WhisperModel("large-v3", compute_type="int8_float16", device="cuda", device_index=0)

print("Loading TTS model (Orpheus 3B)...")
tts_engine = OrpheusTTS()   # downloads canopylabs/orpheus-3b-0.1-ft on first run

print("Models loaded. Starting server...")


# ── STT ───────────────────────────────────────────────────────────────────────
def transcribe(audio_bytes: bytes) -> str:
    """Transcribe raw audio bytes (any format soundfile can read) → text."""
    audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    segments, _ = stt_model.transcribe(audio_array, language="en")
    return "".join(s.text for s in segments).strip()


# ── LLM ───────────────────────────────────────────────────────────────────────
async def stream_llm(prompt: str):
    """Async generator yielding LLM tokens from Ollama streaming API."""
    import aiohttp
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload) as resp:
            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                if data.get("done"):
                    break


# ── TTS ───────────────────────────────────────────────────────────────────────
def synthesize_to_wav_b64(text: str) -> str:
    """Synthesize one sentence → base64-encoded WAV string (24kHz mono)."""
    chunks = list(tts_engine.generate_speech(text, voice=TTS_VOICE))
    audio = np.concatenate(chunks).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


# ── Sentence Splitter ─────────────────────────────────────────────────────────
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

def split_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Split completed sentences out of a running buffer.
    Returns (list_of_complete_sentences, remaining_buffer).
    """
    parts = SENTENCE_END.split(buffer)
    if len(parts) == 1:
        return [], buffer          # no sentence boundary yet
    complete = parts[:-1]
    remainder = parts[-1]
    return complete, remainder


# ── Main Handler ──────────────────────────────────────────────────────────────
async def handle_client(ws):
    print(f"Client connected: {ws.remote_address}")
    try:
        async for message in ws:
            if not isinstance(message, bytes):
                continue   # ignore non-binary frames

            t0 = time.time()

            # 1. STT
            try:
                transcript = transcribe(message)
            except Exception as e:
                await ws.send(json.dumps({"type": "error", "message": f"STT error: {e}"}))
                continue

            if not transcript:
                continue

            await ws.send(json.dumps({"type": "transcript", "text": transcript}))
            print(f"STT ({time.time()-t0:.2f}s): {transcript}")

            # 2. LLM + 3. TTS (interleaved)
            buffer = ""
            async for token in stream_llm(transcript):
                await ws.send(json.dumps({"type": "token", "text": token}))
                buffer += token

                sentences, buffer = split_sentences(buffer)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    print(f"  TTS: {sentence[:60]}...")
                    wav_b64 = await asyncio.get_event_loop().run_in_executor(
                        None, synthesize_to_wav_b64, sentence
                    )
                    await ws.send(json.dumps({"type": "audio", "data": wav_b64}))

            # flush any remaining text in buffer
            if buffer.strip():
                wav_b64 = await asyncio.get_event_loop().run_in_executor(
                    None, synthesize_to_wav_b64, buffer.strip()
                )
                await ws.send(json.dumps({"type": "audio", "data": wav_b64}))

            await ws.send(json.dumps({"type": "done"}))
            print(f"Turn done ({time.time()-t0:.2f}s total)")

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {ws.remote_address}")


# ── Entry Point ───────────────────────────────────────────────────────────────
async def main():
    async with websockets.serve(handle_client, "0.0.0.0", WS_PORT):
        print(f"Voice agent listening on ws://0.0.0.0:{WS_PORT}")
        await asyncio.Future()   # run forever

if __name__ == "__main__":
    asyncio.run(main())
```

**Note:** `aiohttp` is needed for async Ollama streaming — add to requirements: `aiohttp>=3.9`.

---

## index.html — Full Implementation Spec

Single HTML file, no build step. Key behaviour:

### Layout
```
┌─────────────────────────────────┐
│  Voice Agent                    │
│                                 │
│  [conversation history]         │
│   You: ...                      │
│   Agent: ...                    │
│                                 │
│  [● Hold to speak]   Status:    │
└─────────────────────────────────┘
```

### JavaScript Logic

```javascript
// 1. Connect WebSocket on page load
const ws = new WebSocket("ws://localhost:8765");

// 2. Hold mic button (or Space) → MediaRecorder starts (audio/webm;codecs=opus)
// 3. Release → stop recording → send blob as binary WebSocket frame
//    ws.send(audioBlob)  ← send as Blob directly (binary frame)

// 4. Handle incoming messages:
ws.onmessage = async (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "transcript") {
        appendMessage("You", msg.text);
    }
    if (msg.type === "token") {
        appendAgentToken(msg.text);   // stream tokens into agent bubble
    }
    if (msg.type === "audio") {
        const wav = base64ToArrayBuffer(msg.data);
        await audioQueue.enqueue(wav);  // plays in sequence, non-overlapping
    }
    if (msg.type === "done") {
        setStatus("Ready");
    }
    if (msg.type === "error") {
        setStatus("Error: " + msg.message);
    }
};

// Audio queue: decode + play WAV chunks in order without overlap
class AudioQueue {
    constructor() { this.queue = []; this.playing = false; }
    async enqueue(arrayBuffer) {
        const audioCtx = getAudioContext();  // reuse single AudioContext
        const decoded = await audioCtx.decodeAudioData(arrayBuffer);
        this.queue.push(decoded);
        if (!this.playing) this._playNext();
    }
    async _playNext() {
        if (!this.queue.length) { this.playing = false; return; }
        this.playing = true;
        const buf = this.queue.shift();
        const src = audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(audioCtx.destination);
        src.onended = () => this._playNext();
        src.start();
    }
}
```

---

## VRAM at Runtime

| Component                   | VRAM      |
|-----------------------------|-----------|
| faster-whisper large-v3 INT8 | ~2–3 GB  |
| Qwen2.5 14B Q4_K_M (Ollama) | ~9 GB    |
| Orpheus 3B FP16             | ~6–7 GB   |
| **Total**                   | **~17–19 GB** |
| **Headroom on 3090 (24 GB)**| **~5–7 GB** |

---

## Startup Sequence

```bash
# Terminal 1 — LLM (already running, just in case)
ollama serve

# Terminal 2 — Voice pipeline
cd ~/Desktop/Voice\ agents/Pipeline
/home/nauyan/miniconda3/bin/python3 -u server.py
# prints "Models loaded. Starting server..." after ~30s (first run downloads Orpheus)

# Browser — UI
# Open index.html directly, or:
python3 -m http.server 3000   # then http://localhost:3000
```

---

## Latency Expectations

| Stage | Expected |
|-------|----------|
| STT (faster-whisper INT8) | 50–80 ms |
| LLM time-to-first-token | 100–200 ms |
| Orpheus first sentence (GPU) | 80–150 ms |
| **Total TTFA** | **~250–450 ms** |

---

## Verification Checklist

- [ ] `server.py` starts, prints "Models loaded"
- [ ] `nvidia-smi` shows ~17-19 GB used
- [ ] Browser connects (status shows "Connected")
- [ ] Hold mic → speak → release → transcript appears
- [ ] Agent text streams in token by token
- [ ] Audio plays back, sentence by sentence, no overlap
- [ ] Second turn works (no memory leak / crash)
