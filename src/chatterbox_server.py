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
