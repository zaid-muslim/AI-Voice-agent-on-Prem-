#!/usr/bin/env python3
"""One-shot: synthesize a reference voice clip with Orpheus for Chatterbox Turbo's
voice-cloning prompt. Run once, then this process exits and frees the GPU."""
import os

import numpy as np
import soundfile as sf
from orpheus_tts import OrpheusModel
from vllm import AsyncEngineArgs, AsyncLLMEngine

TEXT = (
    "Hello, this is a short reference recording. "
    "I am speaking clearly and naturally so that the voice cloning system "
    "has enough audio to work with. The quick brown fox jumps over the lazy dog."
)
SAMPLE_RATE = 24000

# Write the clip into the shared voice-seed asset dir (project root / assets/voice_seed).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "assets", "voice_seed", "reference_voice.wav")


class BudgetedOrpheusModel(OrpheusModel):
    def _setup_engine(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name, dtype=self.dtype,
            gpu_memory_utilization=0.5, max_model_len=2048,
        )
        return AsyncLLMEngine.from_engine_args(engine_args)


if __name__ == "__main__":
    print("Loading Orpheus...")
    tts = BudgetedOrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
    print("Synthesizing reference clip...")
    chunks = list(tts.generate_speech(prompt=TEXT, voice="tara"))
    audio = np.frombuffer(b"".join(chunks), dtype=np.int16)
    sf.write(OUTPUT_PATH, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"Saved {OUTPUT_PATH} ({audio.size / SAMPLE_RATE:.2f}s)")
