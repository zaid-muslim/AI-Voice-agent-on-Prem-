import sys
import json
import base64
import signal
import threading
import time
import numpy as np
import torch

# CRITICAL: the parent bridge owns this process's lifecycle exclusively
# (it kills us via _cleanup_process on stop/cancel). Ignore SIGINT so a
# terminal Ctrl+C - which the kernel delivers to the whole foreground
# process group - can't crash us mid-CUDA-call with a KeyboardInterrupt
# (that was the qwen_worker.py:168 traceback, and uncontrolled teardown
# there is a likely source of the OOM seen on the following run).
# Redundant with the parent's start_new_session=True, on purpose.
signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    model = None

    # Guards concurrent writes to stdout: the main thread (reading stdin)
    # and the generation thread (streaming audio chunks) can both want to
    # write at the same time. Without this, two JSON lines could interleave
    # into one corrupt line on the client's side.
    stdout_lock = threading.Lock()

    # Tracks which request id is "current" and lets the main loop tell a
    # still-running generation thread to stop early when a new/cancelled
    # request comes in.
    current_req_id = {"id": None}
    cancel_event = threading.Event()
    gen_thread = {"t": None}

    def write_line(obj):
        with stdout_lock:
            sys.stdout.write(json.dumps(obj) + "\n")
            sys.stdout.flush()

    def log_timing(msg):
        # stderr is inherited by the parent process, so these lines land in
        # the normal pipecat logs without touching the stdout JSON protocol.
        print(msg, file=sys.stderr, flush=True)

    def run_generation(req_id, text, language, speaker, chunk_size, t_cmd_received):
        t_thread_start = time.perf_counter()
        startup_gap_ms = (t_thread_start - t_cmd_received) * 1000
        first_chunk_seen = False
        try:
            t_call = time.perf_counter()
            for audio_chunk, sr, _ in model.generate_custom_voice_streaming(
                text=text, language=language, speaker=speaker, chunk_size=chunk_size
            ):
                # Checked once per yielded chunk. Can't abort mid-chunk (the
                # model's internal CUDA step isn't interruptible from
                # Python), but caps post-interruption overrun to ~one chunk.
                if cancel_event.is_set():
                    break

                if not first_chunk_seen:
                    first_chunk_seen = True
                    t_first = time.perf_counter()
                    log_timing(
                        f"[worker timing] id={req_id} text_len={len(text)} "
                        f"cmd_to_thread_start={startup_gap_ms:.1f}ms "
                        f"synthesis_first_chunk={(t_first - t_call) * 1000:.1f}ms "
                        f"total_worker_side={(t_first - t_cmd_received) * 1000:.1f}ms"
                    )

                pcm_bytes = (
                    (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                )
                write_line(
                    {
                        "id": req_id,
                        "audio_b64": base64.b64encode(pcm_bytes).decode("ascii"),
                        "sample_rate": int(sr),
                    }
                )
        except Exception as e:
            write_line({"id": req_id, "error": str(e)})
        finally:
            # Always signal done for this id - finished, errored, or
            # cancelled - so the client's read loop knows to stop.
            write_line({"id": req_id, "done": True})

    # Infinite loop waiting for commands from the main Pipecat pipeline.
    # Generation runs on its own thread, so this loop is never blocked by it.
    for line in sys.stdin:
        if not line.strip():
            continue
        t_cmd_received = time.perf_counter()
        req = json.loads(line)
        action = req.get("action")

        if action == "init":
            try:
                from faster_qwen3_tts import FasterQwen3TTS

                model_id = req.get(
                    "model_id",
                    "/home/nauyan/voice-agent-pipeline/models/Qwen3-TTS-0.6B-custom",
                )

                model = FasterQwen3TTS.from_pretrained(model_id, dtype="bfloat16")

                # Warm-up: run the FULL streaming path a few times so CUDA
                # graph capture and kernel autotuning are paid up-front, not
                # on the user's first real turn. Your logs showed first-chunk
                # time falling from ~750ms (cold) to ~245ms (warm) only after
                # many turns - these full runs get you the ~245ms floor on
                # turn one. Each run consumes ALL chunks (not a bare break)
                # so every kernel in the path is actually exercised.
                t_warmup_start = time.perf_counter()
                cancel_event.clear()
                for _ in range(3):
                    for _chunk in model.generate_custom_voice_streaming(
                        text="Warming up the speech synthesis engine now.",
                        language=req.get("language", "English"),
                        speaker=req.get("speaker", "aiden"),
                        chunk_size=req.get("chunk_size", 8),
                    ):
                        pass  # drain fully

                log_timing(
                    f"[worker timing] warm-up (3 full runs) took "
                    f"{(time.perf_counter() - t_warmup_start) * 1000:.1f}ms"
                )
                write_line({"status": "ready"})

            except Exception as e:
                log_timing(f"CRITICAL WORKER CRASH DURING INIT: {str(e)}")
                import traceback

                log_timing(traceback.format_exc())
                sys.exit(1)

        elif action == "tts":
            # If a previous generation is still running, cancel it and wait
            # for it to exit before starting a new one.
            if gen_thread["t"] is not None and gen_thread["t"].is_alive():
                t_before_join = time.perf_counter()
                cancel_event.set()
                gen_thread["t"].join()
                join_ms = (time.perf_counter() - t_before_join) * 1000
                log_timing(
                    f"[worker timing] stale generation join took {join_ms:.1f}ms "
                    f"before starting id={req.get('id')}"
                )

            req_id = req.get("id")
            current_req_id["id"] = req_id
            cancel_event.clear()

            t = threading.Thread(
                target=run_generation,
                args=(
                    req_id,
                    req.get("text", ""),
                    req.get("language", "English"),
                    req.get("speaker", "aiden"),
                    req.get("chunk_size", 8),
                    t_cmd_received,
                ),
                daemon=True,
            )
            gen_thread["t"] = t
            t.start()

        elif action == "cancel":
            if req.get("id") == current_req_id["id"]:
                cancel_event.set()


if __name__ == "__main__":
    main()
