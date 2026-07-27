#!/usr/bin/env python
"""
Kokoro TTS (hexgrad/Kokoro-82M) subprocess worker.

REAL API USED HERE (verified by downloading and reading kokoro v0.9.4's
actual wheel source, NOT guessed):
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='a')          # 'a' = American English
    for result in pipeline(text, voice="af_heart", speed=1):
        audio = result.audio                      # torch.FloatTensor, 24000 Hz

LICENSE: Apache-2.0, confirmed directly on the model card
(hexgrad/Kokoro-82M) - commercial use is explicitly welcomed by the
maintainer, unlike some other open-weight TTS models. Small (82M params,
~2GB VRAM) - the lightest-weight option in this project's TTS lineup.

HONEST ARCHITECTURAL NOTE - THE GOOD KIND, FOR ONCE: unlike Chatterbox's
generate() (one monolithic call, full utterance before any audio exists),
Kokoro's __call__ is a REAL Python generator that yields one Result per
text segment AS IT'S PRODUCED - genuine incremental synthesis, not just
chunked transport of an already-finished waveform. This worker streams
each segment's audio out the moment it's ready, so "first chunk" timing
here is closer to Qwen's true head-start behavior than Chatterbox's.
Still worth measuring for real via the dev console's latency comparison
rather than assuming - that's the whole point of that feature.

VOICES: Kokoro ships multiple named voice packs (e.g. "af_heart",
"af_bella", "am_adam" - see hexgrad/Kokoro-82M's VOICES.md for the full
list). The `model` field in system_config.json's tts.model IS the voice
name for this engine (not a checkpoint path - Kokoro-82M is a single
model, only the voice pack varies).

STDOUT DISCIPLINE (found the hard way, live): this worker's stdin/stdout
is a strict line-delimited JSON protocol - the LiveKit bridge
(plugins/kokoro_tts.py) reads stdout expecting EVERY line to be exactly
one JSON message. During init, some dependency in the load path (torch /
misaki / huggingface_hub / espeak - exact source not pinned down, likely
varies by version) can write a stray line directly to real stdout
(progress text, a warning, etc.), NOT through Python's `warnings` or
`logging` module, and NOT through this file's own `_log()` (which
correctly goes to stderr already). That stray line lands in the pipe
BEFORE this worker's own {"ready": true} line, so the bridge - which reads
only the first stdout line as the ready payload - parses garbage instead
and fails with `json.decoder.JSONDecodeError: Expecting value: line 1
column 1 (char 0)`. Real fix: redirect real stdout to stderr for the
ENTIRE model-load/warm-up window, restoring it only to emit our own
protocol lines. This is why `main()` below swaps `sys.stdout` around the
`init` action's model-loading block instead of just trusting that nothing
else writes to it.

NOT YET RUN END-TO-END: protocol logic verified against a stub model
initially; init path (including the stdout-noise issue above) has now
been confirmed against the real model load on real hardware. The `tts`
synthesis path (per-segment streaming during actual speech generation) is
UNCHANGED from the original streaming implementation and has NOT yet been
observed under the same real-hardware scrutiny as init. If a similar
stray stdout write ever happens mid-synthesis (not yet seen, but the same
class of bug as the init one), it would corrupt an audio_b64/done line
the same way - worth watching for specifically on the first real
multi-sentence synthesis call rather than assuming init-only. Deliberately
NOT adding the same stdout-redirect guard here pre-emptively, since doing
so would require buffering all segments before emitting any of them,
which throws away Kokoro's actual streaming advantage over Chatterbox (see
the HONEST ARCHITECTURAL NOTE above) for a problem that hasn't actually
occurred here - fix this path only if/when it's actually observed.

PROTOCOL: identical shape to chatterbox_worker.py/parakeet_worker.py -
{"action":"init"} -> {"ready": true}
{"action":"tts","id":...,"text":...,"voice":...} -> one or more
    {"id":...,"audio_b64":...,"sample_rate":24000} then {"id":...,"done":true}
{"action":"cancel","id":...} -> best-effort, checked between segments -
    genuinely more effective here than in Chatterbox's worker, since
    Kokoro's generator naturally yields control between segments where
    Chatterbox's single generate() call does not.
{"action":"shutdown"} -> exits

STANDALONE SANITY CHECK:
    source ~/voice-agent-pipeline/.venv-kokoro/bin/activate
    printf '{"action": "init"}\n' | python kokoro_worker.py
"""

import base64
import json
import sys
import time
import traceback

import numpy as np

DEFAULT_VOICE = "af_heart"
DEFAULT_LANG_CODE = "a"  # American English
SAMPLE_RATE = 24000


def _log(msg: str) -> None:
    print(f"[kokoro_worker] {msg}", file=sys.stderr, flush=True)


def _tensor_to_pcm16_bytes(tensor_like) -> bytes:
    """Same float->int16 conversion as chatterbox_worker.py - framework-
    agnostic, works on anything array-like with a numpy-compatible shape."""
    arr = np.asarray(tensor_like, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def main() -> None:
    pipeline = None
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
            _log(f"loading Kokoro-82M (lang_code={DEFAULT_LANG_CODE}) ...")
            # CRITICAL: redirect real stdout to stderr for the whole load
            # + warm-up window - see the STDOUT DISCIPLINE note in this
            # file's module docstring for exactly why this is necessary
            # (a stray non-JSON line from a dependency here breaks the
            # bridge's protocol parsing). Restored immediately after, so
            # our own {"ready": ...} line is guaranteed to be the first
            # (and only) thing written to the real stdout pipe.
            real_stdout = sys.stdout
            sys.stdout = sys.stderr
            try:
                from kokoro import KPipeline  # deferred: heavy import

                pipeline = KPipeline(lang_code=DEFAULT_LANG_CODE)
                # Warm-up: one short synthesis so CUDA kernels/graph are
                # compiled before the first real caller.
                for _ in pipeline("Warm up.", voice=DEFAULT_VOICE):
                    pass
                _log(f"ready in {time.time() - t0:.1f}s")
            except Exception as exc:  # noqa: BLE001
                _log(f"INIT FAILED: {exc}\n{traceback.format_exc()}")
                sys.stdout = real_stdout
                print(json.dumps({"ready": False, "error": str(exc)}), flush=True)
                return
            sys.stdout = real_stdout
            print(json.dumps({"ready": True}), flush=True)

        elif action == "tts":
            req_id = cmd.get("id")
            text = cmd.get("text", "")
            voice = cmd.get("voice", DEFAULT_VOICE)
            try:
                if pipeline is None:
                    raise RuntimeError("model not initialized - send 'init' first")

                for result in pipeline(text, voice=voice):
                    if req_id in cancelled_ids:
                        break  # genuine early-exit: Kokoro's generator
                        # yields control between segments, unlike
                        # Chatterbox's one-shot generate()
                    if result.audio is None:
                        continue
                    pcm_bytes = _tensor_to_pcm16_bytes(result.audio)
                    b64 = base64.b64encode(pcm_bytes).decode()
                    print(
                        json.dumps(
                            {"id": req_id, "audio_b64": b64, "sample_rate": SAMPLE_RATE}
                        ),
                        flush=True,
                    )
                cancelled_ids.discard(req_id)
                print(json.dumps({"id": req_id, "done": True}), flush=True)
            except Exception as exc:  # noqa: BLE001
                _log(f"tts error (id={req_id}): {exc}\n{traceback.format_exc()}")
                print(json.dumps({"id": req_id, "error": str(exc)}), flush=True)

        elif action == "cancel":
            cancelled_ids.add(cmd.get("id"))

        elif action == "shutdown":
            _log("shutdown requested, exiting")
            return

        else:
            _log(f"unknown action: {action!r}")


if __name__ == "__main__":
    main()
