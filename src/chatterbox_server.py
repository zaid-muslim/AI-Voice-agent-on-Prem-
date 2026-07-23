#!/usr/bin/env python3

import asyncio
import io
import os

import soundfile as sf
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fastapi import FastAPI, Response
from pydantic import BaseModel

# Anchor asset paths to the project root (parent of src/) so they resolve no
# matter what directory the process is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Filename only (resolved against assets/voice_seed/) — not a full path — so a pool of instances
# can each be given a different voice via one env var, picking from whatever .wav files are baked
# into the image (see Dockerfile.chatterbox, which now copies the whole voice_seed/ directory
# rather than a single hardcoded file).
VOICE_FILE = os.environ.get("CHATTERBOX_VOICE_FILE", "reference_trump.wav")
REFERENCE_AUDIO = os.path.join(PROJECT_ROOT, "assets", "voice_seed", VOICE_FILE)
# 0.85 = more expressive delivery (library default 0.5; >0.9 tends to distort) — the value already
# tuned for reference_trump.wav specifically. A different reference clip may want different tuning,
# hence configurable rather than hardcoded now that voice varies per instance.
EXAGGERATION = float(os.environ.get("CHATTERBOX_EXAGGERATION", "0.85"))
PORT = 8766

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Chatterbox Turbo on {device}...")
model = ChatterboxTurboTTS.from_pretrained(device=device)
print(f"Priming voice conditioning from reference clip ({VOICE_FILE}, exaggeration={EXAGGERATION})...")
model.prepare_conditionals(REFERENCE_AUDIO, exaggeration=EXAGGERATION)
print("Chatterbox Turbo ready.")

# One model instance, shared by every caller (single voice pipeline, now serving multiple
# concurrent LiveKit rooms) — model.generate() is NOT safe to call from two requests at once
# (shared internal state, single CUDA context), and FastAPI would otherwise happily run
# overlapping requests in its threadpool. This lock forces synthesis requests to queue instead
# of racing — a caller's audio may arrive a bit later under concurrent load, but never garbled.
_synth_lock = asyncio.Lock()

# Note: two bf16 approaches were benchmarked here and both measured SLOWER than fp32, not faster:
#   - bare torch.autocast around generate(): 16-115% slower (casts activations on the fly, master
#     weights stay fp32 — adds per-op cast overhead with none of the memory-bandwidth win).
#   - converting model.t3's weights to bf16 (+ autocast for the leftover fp32 conditioning
#     tensors): still 4-35% slower. The "batch=1 decode is memory-bandwidth-bound, bf16 helps"
#     heuristic holds for large (7B+) LLMs but didn't transfer to this ~400M-param T3 backbone on
#     this GPU/library stack — fixed overhead (kernel launch, autocast's per-op dtype checks
#     across ~100-170 sequential steps) evidently dominates over the bandwidth saved. Left as
#     fp32-only; don't re-attempt this without new evidence the overhead profile has changed.


class SynthesizeRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict[str, str]:
    """Readiness probe (Docker HEALTHCHECK / orchestrator.py's HTTP poll). The module-scope
    model load above already blocked until ready, so a served response here means synthesis is
    ready — no separate readiness state to track, same contract as the whisper service's /health."""
    return {"status": "ok"}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    loop = asyncio.get_running_loop()
    async with _synth_lock:
        # run_in_executor (not a bare blocking call) so the lock doesn't also stall the event
        # loop for unrelated requests (health checks, etc.) while this synthesis runs.
        wav = await loop.run_in_executor(None, model.generate, req.text)
    buf = io.BytesIO()
    sf.write(buf, wav.squeeze(0).numpy(), model.sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
