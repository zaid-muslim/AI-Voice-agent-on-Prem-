#!/usr/bin/env python3

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
REFERENCE_AUDIO = os.path.join(PROJECT_ROOT, "assets", "voice_seed", "reference_trump.wav")
PORT = 8766

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Chatterbox Turbo on {device}...")
model = ChatterboxTurboTTS.from_pretrained(device=device)
print("Priming voice conditioning from reference clip...")
model.prepare_conditionals(REFERENCE_AUDIO, exaggeration=0.85)  # more expressive delivery (default 0.5; >0.9 tends to distort)
print("Chatterbox Turbo ready.")

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


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    wav = model.generate(req.text)
    buf = io.BytesIO()
    sf.write(buf, wav.squeeze(0).numpy(), model.sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
