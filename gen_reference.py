#!/usr/bin/env python3
"""One-shot: synthesize a reference voice clip with Orpheus for Chatterbox Turbo's
voice-cloning prompt. Run once, then this process exits and frees the GPU."""
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
    sf.write("reference_voice.wav", audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"Saved reference_voice.wav ({audio.size / SAMPLE_RATE:.2f}s)")
