#!/usr/bin/env python
"""
Piper TTS (OHF-Voice/piper1-gpl) subprocess worker.

REAL API USED HERE (verified by downloading and reading piper-tts v1.5.0's
actual wheel source, NOT guessed):
    from piper import PiperVoice
    voice = PiperVoice.load(model_path, config_path=None, use_cuda=False)
    for audio_chunk in voice.synthesize(text):
        audio_chunk.sample_rate    # int, defined by the voice model
        audio_chunk.audio_int16_bytes  # already raw PCM16 bytes - no
                                        # conversion needed, unlike every
                                        # other TTS worker in this project

*** REAL LICENSE CAVEAT - READ BEFORE USING THIS IN A COMMERCIAL PRODUCT ***
The confirmed package metadata shows License: GPL-3.0-or-later (the
actively maintained fork, OHF-Voice/piper1-gpl - the older MIT-licensed
rhasspy/piper repo is archived/read-only as of Oct 2025 and its PyPI
package is unmaintained). GPL-3.0 has real copyleft implications that MIT
did not. This is NOT a code-quality issue - it's a legal one, and it's
YOUR call to make, not something to route around in code. Get this
checked against your actual deployment/distribution model before shipping
Piper in anything commercial. It stays in this project as a real, working,
CPU-friendly option, with this caveat surfaced clearly rather than buried.

WHY IT'S HERE ANYWAY: unlike every other TTS in this project, Piper needs
NO GPU and NO torch at all for inference (only `onnxruntime`) - genuinely
useful as a fallback that works even if your GPU is fully saturated by the
LLM + STT, or on a machine with no GPU at all. That's a real, distinct
value proposition worth having available, GPL caveat notwithstanding.

VOICE MODELS: Piper needs a `.onnx` model file AND its matching `.json`
config, downloaded separately (not auto-fetched via a from_pretrained()
call the way every other engine in this project works) - see
https://github.com/rhasspy/piper/blob/master/VOICES.md for the full voice
list and https://huggingface.co/rhasspy/piper-voices for direct
downloads. system_config.json's tts.model field for this engine is the
PATH to the .onnx file (config is assumed to be model_path + ".json",
Piper's own default convention per its real load() signature).

NOT YET RUN END-TO-END: protocol logic verified against a stub voice
object (same rigor as every other worker in this project).

PROTOCOL: same shape as the other TTS workers, with one addition - "model"
in the init command tells the worker WHICH .onnx voice file to load
(Piper is CPU-fast enough that loading a specific voice per-worker,
rather than switching voices within one running worker, is the simpler,
proven-pattern choice here):
{"action":"init","model_path":"/path/to/voice.onnx","use_cuda":false}
    -> {"ready": true}
{"action":"tts","id":...,"text":...}
    -> one or more {"id":...,"audio_b64":...,"sample_rate":...}
       then {"id":...,"done":true}
{"action":"shutdown"} -> exits

STANDALONE SANITY CHECK:
    source ~/voice-agent-pipeline/.venv-piper/bin/activate
    printf '{"action": "init", "model_path": "/path/to/en_US-lessac-medium.onnx"}\n' \
        | python piper_worker.py
"""

import base64
import json
import sys
import time
import traceback

DEFAULT_MODEL_PATH = (
    "/home/nauyan/voice-agent-pipeline/models/piper/en_US-lessac-medium.onnx"
)


def _log(msg: str) -> None:
    print(f"[piper_worker] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    voice = None

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
            model_path = cmd.get("model_path", DEFAULT_MODEL_PATH)
            use_cuda = bool(cmd.get("use_cuda", False))
            _log(f"loading Piper voice from {model_path} (use_cuda={use_cuda}) ...")
            try:
                from piper import PiperVoice  # deferred import

                voice = PiperVoice.load(model_path, use_cuda=use_cuda)
                # Warm-up: one short synthesis, cheap on CPU too but still
                # worth paying before the first real caller.
                for _ in voice.synthesize("Warm up."):
                    pass
                _log(f"ready in {time.time() - t0:.1f}s")
                print(json.dumps({"ready": True}), flush=True)
            except Exception as exc:  # noqa: BLE001
                _log(f"INIT FAILED: {exc}\n{traceback.format_exc()}")
                print(json.dumps({"ready": False, "error": str(exc)}), flush=True)
                return

        elif action == "tts":
            req_id = cmd.get("id")
            text = cmd.get("text", "")
            try:
                if voice is None:
                    raise RuntimeError("model not initialized - send 'init' first")

                sample_rate = None
                for audio_chunk in voice.synthesize(text):
                    sample_rate = audio_chunk.sample_rate
                    # Piper's AudioChunk already carries raw int16 PCM
                    # bytes directly - no float->int16 conversion step
                    # needed here, unlike every other TTS worker in this
                    # project (Chatterbox/Kokoro/Qwen all hand back
                    # floating-point tensors that need converting first).
                    b64 = base64.b64encode(audio_chunk.audio_int16_bytes).decode()
                    print(
                        json.dumps(
                            {"id": req_id, "audio_b64": b64, "sample_rate": sample_rate}
                        ),
                        flush=True,
                    )
                print(json.dumps({"id": req_id, "done": True}), flush=True)
            except Exception as exc:  # noqa: BLE001
                _log(f"tts error (id={req_id}): {exc}\n{traceback.format_exc()}")
                print(json.dumps({"id": req_id, "error": str(exc)}), flush=True)

        elif action == "shutdown":
            _log("shutdown requested, exiting")
            return

        else:
            _log(f"unknown action: {action!r}")


if __name__ == "__main__":
    main()
