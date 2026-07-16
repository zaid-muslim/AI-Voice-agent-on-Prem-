#!/usr/bin/env python
"""
NVIDIA Parakeet TDT subprocess worker.

WHY THIS IS A SEPARATE PROCESS IN A SEPARATE VENV:
nemo_toolkit[asr]'s dependency chain (librosa -> numba -> llvmlite) still
has old pins deep in its resolution graph that have no prebuilt wheel for
Python 3.12 and fail to build from source. Python 3.10/3.11 have real
wheels for the whole chain. Rather than fight that in the main agent's
venv, Parakeet runs HERE, in its own Python 3.10/3.11 venv, talking to the
main LiveKit agent over stdin/stdout JSON - the exact same protocol shape
as qwen_worker.py, for the exact same reason: total dependency isolation.
The main agent's venv never needs nemo/torch/lightning installed at all.

PROTOCOL (JSON, one object per line):
  stdin  {"action": "init"}
  stdout {"ready": true}                              (after model load + warm run)

  stdin  {"action": "transcribe", "id": "...", "audio_b64": "...",
          "sample_rate": 16000}
  stdout {"id": "...", "text": "...", "done": true}
  stdout {"id": "...", "error": "..."}                 (on failure)

  stdin  {"action": "shutdown"}
  -> process exits

STANDALONE SANITY CHECK (confirms nemo loads + this venv is healthy,
without needing the main agent running at all):

    source ~/voice-agent-pipeline/.venv-parakeet/bin/activate
    printf '{"action": "init"}\n' | python parakeet_worker.py

Should print progress to stderr, then {"ready": true} to stdout. First run
downloads ~2.4 GB from Hugging Face.
"""

import base64
import json
import sys
import time
import traceback

import numpy as np


def _log(msg: str) -> None:
    # stderr only - stdout is reserved for the JSON protocol, exactly like
    # qwen_worker.py's convention.
    print(f"[parakeet_worker] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    model = None
    torch = None

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
            _log("loading nvidia/parakeet-tdt-0.6b-v2 ...")
            try:
                import nemo.collections.asr as nemo_asr
                import torch as _torch

                torch = _torch
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name="nvidia/parakeet-tdt-0.6b-v2"
                )
                model.eval()
                if torch.cuda.is_available():
                    model = model.cuda()
                # Warm run: pay CUDA kernel/graph compile cost here, not on
                # the first real caller.
                model.transcribe(
                    [np.zeros(8000, dtype=np.float32)], batch_size=1, verbose=False
                )
                _log(f"ready in {time.time() - t0:.1f}s")
                print(json.dumps({"ready": True}), flush=True)
            except Exception as exc:  # noqa: BLE001
                _log(f"INIT FAILED: {exc}\n{traceback.format_exc()}")
                print(json.dumps({"ready": False, "error": str(exc)}), flush=True)
                return

        elif action == "transcribe":
            req_id = cmd.get("id")
            try:
                if model is None:
                    raise RuntimeError("model not initialized - send 'init' first")
                audio_bytes = base64.b64decode(cmd["audio_b64"])
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
                with torch.inference_mode():
                    out = model.transcribe([audio], batch_size=1, verbose=False)
                first = out[0] if out else ""
                # NeMo's transcribe() return type drifted across versions:
                # list[str] (older) vs list[Hypothesis] with .text (newer).
                text = (
                    first
                    if isinstance(first, str)
                    else getattr(first, "text", str(first))
                )
                print(
                    json.dumps(
                        {"id": req_id, "text": (text or "").strip(), "done": True}
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                _log(f"transcribe error (id={req_id}): {exc}\n{traceback.format_exc()}")
                print(json.dumps({"id": req_id, "error": str(exc)}), flush=True)

        elif action == "shutdown":
            _log("shutdown requested, exiting")
            return

        else:
            _log(f"unknown action: {action!r}")


if __name__ == "__main__":
    main()
