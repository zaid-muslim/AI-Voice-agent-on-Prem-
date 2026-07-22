#!/usr/bin/env python
"""
NVIDIA Canary (canary-180m-flash) subprocess worker.

LICENSE - READ BEFORE PICKING A DIFFERENT CANARY VARIANT: this worker
defaults to nvidia/canary-180m-flash, licensed CC-BY-4.0 (commercial use
explicitly fine). The base nvidia/canary-1b model is CC-BY-NC-4.0 -
NON-COMMERCIAL - confirmed directly on its Hugging Face model card. If you
ever change the model name in system_config.json's stt.model field for
this engine, check the license of whatever you're switching to; don't
assume all Canary variants share the same terms, because they don't.

REAL API USED HERE (verified against nvidia/canary-180m-flash's actual
published usage, NOT guessed - this is a DIFFERENT model class than
Parakeet, not just a different checkpoint name):
    from nemo.collections.asr.models import EncDecMultiTaskModel
    model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-180m-flash')
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    output = model.transcribe(['path.wav'], batch_size=16, pnc='True')
    text = output[0].text

HONEST IMPLEMENTATION CHOICE: Canary's documented transcribe() examples
all pass FILE PATHS, unlike Parakeet's ASRModel.transcribe() which this
project already confirmed accepts raw numpy arrays directly (see
parakeet_worker.py). Canary uses a DIFFERENT model class
(EncDecMultiTaskModel, built for multi-task ASR+translation, not
ASRModel) and its raw-array support was NOT verified in this project's
research pass. Rather than assume feature parity with Parakeet's
different class, this worker writes each utterance to a short-lived temp
WAV file and transcribes via path - slower than a hypothetical raw-array
path (a real, disk-write cost paid per utterance), but CORRECT per the
verified, documented API rather than a guess. If you verify raw-array
support later, this is the specific place to optimize.

SAME NeMo VENV AS PARAKEET: uses the identical nemo_toolkit[asr] install -
if your Parakeet venv (.venv-parakeet from earlier this week) is set up
and working, canary-180m-flash should load in that SAME venv with no
additional dependency work. Confirm with the standalone check below.

PROTOCOL: identical shape to parakeet_worker.py.
  {"action": "init"} -> {"ready": true}
  {"action": "transcribe", "id":..., "audio_b64":..., "sample_rate": 16000}
      -> {"id":..., "text":..., "done": true}
  {"action": "shutdown"} -> exits

STANDALONE SANITY CHECK (same venv as Parakeet):
    source ~/voice-agent-pipeline/.venv-parakeet/bin/activate
    printf '{"action": "init"}\n' | python canary_worker.py
"""

import base64
import json
import os
import sys
import tempfile
import time
import traceback
import wave

import numpy as np

DEFAULT_MODEL = "nvidia/canary-180m-flash"  # CC-BY-4.0, commercial-safe -
# NOT nvidia/canary-1b (CC-BY-NC-4.0, non-commercial) - see module docstring
CANARY_SAMPLE_RATE = 16000


def _log(msg: str) -> None:
    print(f"[canary_worker] {msg}", file=sys.stderr, flush=True)


def _write_temp_wav(audio_float32: np.ndarray, sample_rate: int) -> str:
    """Canary's verified, documented API takes file paths (see module
    docstring's honest note on why raw-array support wasn't assumed).
    This writes a short-lived mono 16-bit WAV and returns its path; caller
    is responsible for deleting it."""
    pcm16 = np.clip(audio_float32, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


def main() -> None:
    model = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            _log(f"bad JSON line, ignoring: {line!r}")
            continue

        action = cmd.get("action")

        if action == "init":
            t0 = time.time()
            model_name = cmd.get("model", DEFAULT_MODEL)
            _log(f"loading {model_name} (EncDecMultiTaskModel) ...")
            try:
                from nemo.collections.asr.models import EncDecMultiTaskModel
                import torch

                model = EncDecMultiTaskModel.from_pretrained(model_name)
                decode_cfg = model.cfg.decoding
                decode_cfg.beam.beam_size = 1
                model.change_decoding_strategy(decode_cfg)
                if torch.cuda.is_available():
                    model = model.cuda()

                # Warm-up: a tiny real WAV, exercising the exact same
                # temp-file code path real requests use.
                warm_path = _write_temp_wav(
                    np.zeros(8000, dtype=np.float32), CANARY_SAMPLE_RATE
                )
                try:
                    model.transcribe([warm_path], batch_size=1, pnc="True")
                finally:
                    os.remove(warm_path)

                _log(f"ready in {time.time() - t0:.1f}s")
                print(json.dumps({"ready": True}), flush=True)
            except Exception as exc:  # noqa: BLE001
                _log(f"INIT FAILED: {exc}\n{traceback.format_exc()}")
                print(json.dumps({"ready": False, "error": str(exc)}), flush=True)
                return

        elif action == "transcribe":
            req_id = cmd.get("id")
            wav_path = None
            try:
                if model is None:
                    raise RuntimeError("model not initialized - send 'init' first")
                audio_bytes = base64.b64decode(cmd["audio_b64"])
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
                sample_rate = cmd.get("sample_rate", CANARY_SAMPLE_RATE)

                wav_path = _write_temp_wav(audio, sample_rate)
                output = model.transcribe([wav_path], batch_size=1, pnc="True")
                text = output[0].text if output else ""
                print(
                    json.dumps(
                        {"id": req_id, "text": (text or "").strip(), "done": True}
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                _log(f"transcribe error (id={req_id}): {exc}\n{traceback.format_exc()}")
                print(json.dumps({"id": req_id, "error": str(exc)}), flush=True)
            finally:
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)

        elif action == "shutdown":
            _log("shutdown requested, exiting")
            return

        else:
            _log(f"unknown action: {action!r}")


if __name__ == "__main__":
    main()
