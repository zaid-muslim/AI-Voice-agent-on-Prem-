"""
benchmark_latency.py - measures REAL per-stage wall-clock latency of the
voice pipeline's hot path, component by component, on whatever of this
machine's own models/services are actually reachable at run time.

WHY THIS EXISTS: the README's "Results" section reports numbers from THIS
script, not estimates - re-run it after any latency-relevant change (a
different STT/TTS engine, a different endpointing config, different
hardware) instead of trusting stale numbers. It never fabricates a value
for a component it couldn't reach; it prints "not reachable" and says why.

Not a pytest file (needs a GPU + locally-installed models it can't assume
CI has) - run by hand:
    python tests/manual/benchmark_latency.py
Optionally point it at a running vLLM / shared-STT / shared-TTS / Qwen-Omni
instance via the same env vars main.py uses (VLLM_BASE_URL,
SHARED_STT_BASE_URL, SHARED_TTS_BASE_URL, QWEN_OMNI_BASE_URL) - each stage
is independent and skips cleanly if its service isn't up.
"""

from __future__ import annotations

import io
import json
import os
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app"
sys.path.insert(0, str(APP_DIR))

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
SHARED_STT_BASE_URL = os.environ.get("SHARED_STT_BASE_URL", "http://localhost:8020")
QWEN_OMNI_BASE_URL = os.environ.get("QWEN_OMNI_BASE_URL", "http://192.168.18.56:8091/v1")
LLM_MODEL = os.environ.get("BENCH_LLM_MODEL", "gemma-4-12b")

N_RUNS = 8  # per stage, after 1 discarded warm-up run


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _summary(samples_s: list[float]) -> dict:
    ms = [s * 1000 for s in samples_s]
    return {
        "n": len(ms),
        "mean_ms": round(statistics.mean(ms), 1),
        "p50_ms": round(statistics.median(ms), 1),
        "min_ms": round(min(ms), 1),
        "max_ms": round(max(ms), 1),
    }


def _synthetic_utterance_wav(seconds: float = 3.0, sample_rate: int = 16000) -> bytes:
    """A synthetic ~3s clip (speech-shaped noise, not real speech) used ONLY
    to measure processing latency, not transcription accuracy - whisper's
    decode cost is driven by audio duration/mel-frame count, not content,
    so this is a fair proxy for "how long does N seconds of caller audio
    take to process", clearly labeled as such wherever it's reported."""
    rng = np.random.default_rng(0)
    n = int(seconds * sample_rate)
    # Band-limited noise (roughly voice-band) rather than white noise/silence,
    # so the STT engine's VAD/feature path isn't handed a degenerate input.
    t = np.arange(n) / sample_rate
    signal = 0.05 * rng.standard_normal(n)
    for freq in (150, 400, 900):
        signal += 0.03 * np.sin(2 * np.pi * freq * t)
    audio = np.clip(signal, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue(), pcm16, sample_rate


def bench_vad() -> dict | None:
    """Silero VAD: per-frame forward-pass latency, same model/loading path
    prewarm() in main.py uses. Must run inside the event loop VADStream
    creates its internal tasks on."""
    try:
        from livekit.plugins import silero
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"silero plugin unavailable: {exc}"}

    async def _run():
        import asyncio

        from livekit import rtc

        vad = silero.VAD.load(min_silence_duration=0.4)
        _, pcm16, sr = _synthetic_utterance_wav(seconds=1.0)
        frame_len = int(sr * 0.032)
        samples = []
        for _ in range(N_RUNS + 1):
            stream = vad.stream()
            start = time.perf_counter()
            for i in range(0, len(pcm16) - frame_len, frame_len):
                chunk = pcm16[i : i + frame_len]
                stream.push_frame(
                    rtc.AudioFrame(
                        data=chunk.tobytes(),
                        sample_rate=sr,
                        num_channels=1,
                        samples_per_channel=len(chunk),
                    )
                )
            stream.end_input()
            async for _event in stream:
                pass  # drain to completion; timing includes full 1s of audio
            samples.append(time.perf_counter() - start)
            await stream.aclose()
        return samples

    try:
        import asyncio

        samples = asyncio.run(_run())[1:]  # discard warm-up
        return _summary(samples) | {"unit": "wall-clock to process 1s of audio"}
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"VAD benchmark failed: {exc}"}


def bench_turn_detector() -> dict | None:
    """inference.TurnDetector(version='v1-mini'): the new API is
    AUDIO-native (not chat-text like the old EnglishModel it replaces) -
    push raw audio frames, flush, and time predict()'s future to resolve.
    This is itself a notable latency-relevant fact for the README: the
    semantic turn model no longer waits on a finished transcript."""
    try:
        from livekit.agents import inference
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"inference module unavailable: {exc}"}

    async def _run():
        from livekit import rtc

        detector = inference.TurnDetector(version="v1-mini")
        _, pcm16, sr = _synthetic_utterance_wav(seconds=1.5)
        frame_len = int(sr * 0.032)
        samples = []
        for _ in range(N_RUNS + 1):
            stream = detector.stream()
            try:
                for i in range(0, len(pcm16) - frame_len, frame_len):
                    chunk = pcm16[i : i + frame_len]
                    stream.push_audio(
                        rtc.AudioFrame(
                            data=chunk.tobytes(),
                            sample_rate=sr,
                            num_channels=1,
                            samples_per_channel=len(chunk),
                        )
                    )
                start = time.perf_counter()
                fut = stream.predict()  # NOTE: no flush() - it calls
                # cancel_inference() internally, which resolves the very
                # future predict() just returned with a default (non-real)
                # event if called afterward. flush() is for signalling a
                # NEW speech segment (cancelling a stale prior prediction),
                # not for pairing with predict() in the same turn.
                await fut
                samples.append(time.perf_counter() - start)
            finally:
                await stream.aclose()
        return samples

    try:
        import asyncio

        samples = asyncio.run(_run())[1:]
        return _summary(samples)
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"turn detector benchmark failed: {exc}"}


def bench_stt_local_whisper() -> dict | None:
    """faster-whisper distil-large-v3, loaded in-process (same engine the
    shared stt_service/server.py wraps), transcribing a synthetic ~3s clip."""
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"faster-whisper unavailable: {exc}"}

    try:
        model_name = os.environ.get("WHISPER_MODEL", "distil-large-v3")
        model = WhisperModel(model_name, device="cuda", compute_type="int8_float16")
        wav_bytes, _, _ = _synthetic_utterance_wav(seconds=3.0)
        buf = io.BytesIO(wav_bytes)

        samples = []
        for _ in range(N_RUNS + 1):
            buf.seek(0)
            start = time.perf_counter()
            segments, _info = model.transcribe(buf, language="en", beam_size=1)
            list(segments)  # force decode - generator is lazy otherwise
            samples.append(time.perf_counter() - start)
        samples = samples[1:]
        return _summary(samples) | {"model": model_name, "audio_seconds": 3.0}
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"local whisper benchmark failed: {exc}"}


def bench_shared_stt_service() -> dict | None:
    """Same measurement via the shared stt_service/server.py HTTP path, if
    it's running - includes the real HTTP round-trip this project's
    production path actually pays."""
    import urllib.request

    try:
        urllib.request.urlopen(f"{SHARED_STT_BASE_URL}/health", timeout=2)
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"{SHARED_STT_BASE_URL} not reachable: {exc}"}

    import urllib.error

    wav_bytes, _, _ = _synthetic_utterance_wav(seconds=3.0)
    samples = []
    for _ in range(N_RUNS + 1):
        boundary = "----bench"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="file"; filename="clip.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{SHARED_STT_BASE_URL}/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as exc:
            return {"skipped": f"shared STT service returned HTTP {exc.code}"}
        samples.append(time.perf_counter() - start)
    samples = samples[1:]
    return _summary(samples) | {"audio_seconds": 3.0, "via": "http (production path)"}


def bench_llm_ttft() -> dict | None:
    """Real vLLM time-to-first-token for a short, realistic tool-call-style
    prompt, over the actual OpenAI-compatible HTTP path main.py uses."""
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{VLLM_BASE_URL}/models", timeout=3)
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"{VLLM_BASE_URL} not reachable: {exc}"}

    samples = []
    for _ in range(N_RUNS + 1):
        payload = json.dumps(
            {
                "model": LLM_MODEL,
                "stream": True,
                "max_tokens": 60,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a hospital receptionist. Be brief.",
                    },
                    {
                        "role": "user",
                        "content": "I'd like to book a cardiology appointment for next Tuesday.",
                    },
                ],
            }
        ).encode()
        req = urllib.request.Request(
            f"{VLLM_BASE_URL}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                for line in resp:
                    if line.strip() and line.strip() != b"data: [DONE]":
                        samples.append(time.perf_counter() - start)
                        break
        except urllib.error.HTTPError as exc:
            return {"skipped": f"vLLM returned HTTP {exc.code}: {exc.read()[:200]}"}
    samples = samples[1:]
    if not samples:
        return {"skipped": "no successful completions"}
    return _summary(samples) | {"metric": "time-to-first-token (streamed)"}


def bench_tts_ttfb() -> dict | None:
    """Real Qwen3-TTS-via-vLLM-Omni time-to-first-byte, if PC2 (or a local
    fallback at the same URL) is reachable."""
    import urllib.error
    import urllib.request

    samples = []
    for _ in range(N_RUNS + 1):
        payload = json.dumps(
            {
                "model": os.environ.get("QWEN_OMNI_MODEL", ""),
                "input": "One moment, let me check the schedule for you.",
                "voice": os.environ.get("QWEN_OMNI_VOICE", "Aiden"),
                "response_format": "wav",
            }
        ).encode()
        req = urllib.request.Request(
            f"{QWEN_OMNI_BASE_URL}/audio/speech",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp.read(1)  # first byte only
                samples.append(time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001
            return {"skipped": f"{QWEN_OMNI_BASE_URL} not reachable: {exc}"}
    samples = samples[1:]
    return _summary(samples) | {"metric": "time-to-first-byte"}


def bench_shared_qwen_tts() -> dict | None:
    """Real first-frame latency through the shared local Qwen3-TTS service
    (tts_service/server.py), via the actual production LiveKit plugin
    (plugins/shared_qwen_tts.py) - not a raw HTTP call, so this also
    exercises the exact code path main.py uses when
    system_config.json's tts.engine is "qwen_shared"."""
    import asyncio

    from plugins.shared_qwen_tts import SharedQwenTTS

    async def _run():
        tts_client = SharedQwenTTS()
        try:
            samples = []
            for _ in range(N_RUNS + 1):
                stream = tts_client.synthesize("One moment, let me check the schedule for you.")
                start = time.perf_counter()
                first = True
                async for _ev in stream:
                    if first:
                        samples.append(time.perf_counter() - start)
                        first = False
                        # Drain the rest so the connection closes cleanly,
                        # but only time-to-first-frame is what we report.
            return samples[1:]
        finally:
            await tts_client.aclose()

    try:
        samples = asyncio.run(_run())
        if not samples:
            return {"skipped": "no audio frames received"}
        return _summary(samples) | {"metric": "time-to-first-frame"}
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"{os.environ.get('SHARED_TTS_BASE_URL', 'http://localhost:8021')} not reachable: {exc}"}


def main() -> None:
    stages = {
        "vad_silero": bench_vad,
        "turn_detector_v1_mini": bench_turn_detector,
        "stt_local_faster_whisper": bench_stt_local_whisper,
        "stt_shared_service_http": bench_shared_stt_service,
        "llm_ttft_vllm": bench_llm_ttft,
        "tts_ttfb_qwen_omni": bench_tts_ttfb,
        "tts_ttfb_qwen_shared": bench_shared_qwen_tts,
    }
    results = {}
    for name, fn in stages.items():
        print(f"--- {name} ---", file=sys.stderr)
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            results[name] = {"skipped": f"unhandled error: {exc}"}
        print(json.dumps(results[name], indent=2), file=sys.stderr)

    out_path = Path(__file__).parent / "latency_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
