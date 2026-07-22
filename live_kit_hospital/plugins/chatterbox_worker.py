#!/usr/bin/env python
"""
Chatterbox TTS (Resemble AI) subprocess worker.

WHY ITS OWN VENV: chatterbox-tts pins torch==2.6.0 exactly (per its real
PyPI metadata, checked directly against v0.1.7's wheel). That conflicts
with the main agent venv's torch and with the Parakeet venv's
torch==2.5.1+cu121 - three different pinned torch versions across three
purposes is exactly why this project's pattern is "one dedicated venv per
model with its own pinned deps," same reasoning as plugins/parakeet_worker.py.

REAL API USED HERE (verified by downloading and reading chatterbox-tts
v0.1.7's actual source, NOT guessed):
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device="cuda")   # auto-downloads
                                                            # weights from
                                                            # HF hub
                                                            # (ResembleAI/chatterbox)
                                                            # via hf_hub_download
    model.sr                                               # == 24000 (S3GEN_SR)
    wav_tensor = model.generate(text, ...)                 # torch.Tensor,
                                                            # shape (1, N),
                                                            # float, watermarked

HONEST ARCHITECTURAL DIFFERENCE FROM QWEN: unlike the Qwen worker's
incremental/streamed synthesis, Chatterbox's generate() call produces the
ENTIRE waveform in one shot - there is no partial/incremental output to
stream as it's produced. This worker chunks the FINISHED audio into pieces
purely for transport (so the LiveKit plugin can start playing/sending
audio before the full base64 payload is written to the pipe), but the
time-to-first-chunk here is really time-to-FULL-synthesis, not a true
streaming head-start the way Qwen's worker achieves. Document this
honestly in the plugin's docstring too - don't let the chunked transport
protocol imply a synthesis-level streaming advantage that doesn't exist.

KNOWN INSTALL CAVEAT (real, found via web search of chatterbox's own
GitHub issues): some environments hit a build failure on the `pkuseg`
(spacy-pkuseg) dependency. If `pip install chatterbox-tts` fails on that
package specifically, check that GitHub issue thread (resemble-ai/chatterbox
issue #367) for the current workaround before assuming this worker's code
is at fault.

NOT YET RUN END-TO-END: this worker's protocol logic is verified with a
stub standing in for ChatterboxTTS (same rigor as parakeet_worker.py's
fake-nemo test). The REAL model has NOT been loaded/run here - that needs
real GPU + a real Hugging Face download, neither available in this build
environment. Run the standalone check in this docstring's companion
README section before trusting it live.

PROTOCOL (JSON, one object per line - matches parakeet_worker.py's shape):
  stdin  {"action": "init"}
  stdout {"ready": true}                                  (after model load)

  stdin  {"action": "tts", "id": "...", "text": "...",
          "chunk_size_bytes": 32000}
  stdout {"id": "...", "audio_b64": "...", "sample_rate": 24000}  (one or
                                                            more chunks)
  stdout {"id": "...", "done": true}
  stdout {"id": "...", "error": "..."}                     (on failure)

  stdin  {"action": "cancel", "id": "..."}    (best-effort: sets a flag
          checked between chunk-writes of the CURRENT in-flight request;
          cannot interrupt mid-generate() since that call isn't
          itself interruptible/incremental - see honest note above)

  stdin  {"action": "shutdown"}
  -> process exits

STANDALONE SANITY CHECK (once this venv is set up on the real machine):
    source ~/voice-agent-pipeline/.venv-chatterbox/bin/activate
    printf '{"action": "init"}\n' | python chatterbox_worker.py
Should print progress to stderr, then {"ready": true} to stdout. First run
downloads model weights from Hugging Face (large; be patient, and note
your build environment needs real internet access to huggingface.co,
unlike this development sandbox which does not).
"""

import base64
import json
import sys
import time
import traceback

import numpy as np

DEFAULT_CHUNK_BYTES = 32000  # ~0.67s of 24kHz 16-bit mono audio per chunk


def _log(msg: str) -> None:
    print(f"[chatterbox_worker] {msg}", file=sys.stderr, flush=True)


def _waveform_to_pcm16_bytes(waveform) -> bytes:
    """Convert a float waveform (any array-like with values roughly in
    [-1, 1], e.g. a squeezed torch tensor moved to CPU/numpy, or a plain
    numpy array) into 16-bit PCM bytes - the same wire format
    qwen_tts.py's client expects (TTSAudioRawFrame consumes raw int16 PCM).
    Framework-agnostic: only needs .astype/clip, works identically whether
    the caller passes a numpy array or something that already behaves like
    one (e.g. tensor.cpu().numpy())."""
    arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def main() -> None:
    model = None
    cancelled_ids: set[str] = set()

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
            _log("loading ResembleAI/chatterbox (device=cuda) ...")
            try:
                from chatterbox.tts import ChatterboxTTS  # deferred: heavy import

                model = ChatterboxTTS.from_pretrained(device="cuda")
                # from_pretrained's standard download includes a default
                # conds.pt (per the real from_local() source), so model.conds
                # should already be set - but guard anyway, since generate()
                # without audio_prompt_path requires it and a warm-up
                # failure here shouldn't be a confusing crash.
                if model.conds is not None:
                    model.generate("Warm up.")
                else:
                    _log(
                        "no default voice conditioning found after load - "
                        "skipping warm-up generation; first real call will "
                        "need an audio_prompt_path or will fail with a "
                        "clear 'prepare_conditionals first' error"
                    )
                _log(f"ready in {time.time() - t0:.1f}s")
                print(json.dumps({"ready": True}), flush=True)
            except Exception as exc:  # noqa: BLE001
                _log(f"INIT FAILED: {exc}\n{traceback.format_exc()}")
                print(json.dumps({"ready": False, "error": str(exc)}), flush=True)
                return

        elif action == "tts":
            req_id = cmd.get("id")
            text = cmd.get("text", "")
            chunk_bytes = int(cmd.get("chunk_size_bytes", DEFAULT_CHUNK_BYTES))
            try:
                if model is None:
                    raise RuntimeError("model not initialized - send 'init' first")

                # Full synthesis in one call - see module docstring's honest
                # note: this is NOT incremental/streamed at the model level.
                wav_tensor = model.generate(text)
                pcm_bytes = _waveform_to_pcm16_bytes(
                    wav_tensor.squeeze(0).detach().cpu().numpy()
                )

                if req_id in cancelled_ids:
                    cancelled_ids.discard(req_id)
                    print(json.dumps({"id": req_id, "done": True}), flush=True)
                    continue

                # Chunk the FINISHED audio purely for transport, so the
                # LiveKit-side plugin can start forwarding audio before the
                # entire base64 payload is written to the pipe.
                for start in range(0, len(pcm_bytes), chunk_bytes):
                    if req_id in cancelled_ids:
                        break
                    chunk = pcm_bytes[start : start + chunk_bytes]
                    b64 = base64.b64encode(chunk).decode()
                    print(
                        json.dumps(
                            {"id": req_id, "audio_b64": b64, "sample_rate": model.sr}
                        ),
                        flush=True,
                    )
                cancelled_ids.discard(req_id)
                print(json.dumps({"id": req_id, "done": True}), flush=True)
            except Exception as exc:  # noqa: BLE001
                _log(f"tts error (id={req_id}): {exc}\n{traceback.format_exc()}")
                print(json.dumps({"id": req_id, "error": str(exc)}), flush=True)

        elif action == "cancel":
            # Best-effort: generate() itself can't be interrupted mid-call
            # (see honest note above), so this only takes effect between
            # transport chunks of an ALREADY-finished synthesis, or skips
            # writing chunks for a request whose result arrives after
            # cancellation was requested.
            cancelled_ids.add(cmd.get("id"))

        elif action == "shutdown":
            _log("shutdown requested, exiting")
            return

        else:
            _log(f"unknown action: {action!r}")


if __name__ == "__main__":
    main()
