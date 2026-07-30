"""
benchmark.py - REAL, measured comparison of three ways to turn a caller's
audio into a receptionist reply, all against the SAME live services this
repo already runs (shared vLLM at VLLM_BASE_URL, shared Whisper STT at
SHARED_STT_BASE_URL) - no estimates, no mocks. Mirrors
tests/manual/benchmark_latency.py's conventions: a stage that isn't
reachable is reported as "skipped" with why, never silently faked.

THREE PIPELINES MEASURED, per utterance:
  1. cascade            - the EXISTING production pipeline: shared Whisper
                           STT, then a separate text-only Gemma call. This
                           is what app/main.py runs today, unchanged.
  2. direct_audio_stt    - the pipeline THIS FOLDER'S agent.py actually
                           ships: Gemma transcribes the raw audio directly
                           (GemmaDirectAudioSTT / gemma_audio_client.
                           transcribe_audio), then the SAME kind of
                           text-only Gemma call as the cascade uses the
                           resulting transcript for. No Whisper anywhere.
  3. direct_audio_single_call - the experimental, NOT-yet-wired-into-
                           agent.py path: ONE Gemma call goes straight
                           from raw audio to the final reply
                           (gemma_audio_client.respond_to_audio),
                           skipping the second text-only call entirely.
                           Measured here to show what it could offer;
                           see README.md's "Scoped out for v1" section
                           for why it isn't live yet (tool-calling driven
                           straight off audio needs more validation).

ACCURACY: each fixture's ground-truth text (what was fed to piper to
synthesize it) is known, so transcript quality is checked by simple
word-overlap against that ground truth - not a full WER implementation,
but enough to catch a transcription that's actually wrong, not just
differently punctuated.

CAVEATS this script does NOT cover (see README.md for the honest list):
  - fixtures are clean, synthetic (piper) speech - no real microphone
    noise, accents, or overlapping speech.
  - no concurrent-load test - every call here is serial on an otherwise
    idle GPU, so this does not show what happens under the 3-concurrent-
    caller budget scripts/run_vllm.sh is tuned for.

Run:
    venv/bin/python direct_audio_agent/benchmark.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import gemma_audio_client as gac  # noqa: E402

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
SHARED_STT_BASE_URL = os.environ.get("SHARED_STT_BASE_URL", "http://127.0.0.1:8020")
LLM_MODEL = os.environ.get("BENCH_LLM_MODEL", "gemma-4-12b")
N_RUNS = int(os.environ.get("BENCH_N_RUNS", "3"))  # per utterance, per pipeline

RECEPTIONIST_SYSTEM_PROMPT = (
    "You are a helpful hospital receptionist. Reply briefly to the caller."
)

FIXTURES_DIR = THIS_DIR / "tests" / "fixtures"
UTTERANCES = [
    {
        "name": "short_command",
        "wav": FIXTURES_DIR / "utt_short_command.wav",
        "ground_truth": "Cancel my appointment please.",
    },
    {
        "name": "long_sentence",
        "wav": FIXTURES_DIR / "utt_long_sentence.wav",
        "ground_truth": "I would like to book an appointment with cardiology for tomorrow morning.",
    },
    {
        "name": "name_heavy",
        "wav": FIXTURES_DIR / "utt_name_heavy.wav",
        "ground_truth": (
            "My name is Muhammad Abdullah and I was born on March third "
            "nineteen ninety eight."
        ),
    },
    {
        "name": "general_question",
        "wav": FIXTURES_DIR / "utt_general_question.wav",
        "ground_truth": (
            "What time does the hospital open on Saturdays and is parking available."
        ),
    },
]


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


def _word_overlap_ratio(ground_truth: str, transcript: str) -> float:
    """|intersection| / |ground truth words| - crude but catches an
    actually-wrong transcription, not just punctuation/casing drift."""

    def _words(s: str) -> set[str]:
        return {w.strip(".,!?\"'").lower() for w in s.split() if w.strip(".,!?\"'")}

    gt_words = _words(ground_truth)
    if not gt_words:
        return 1.0
    got_words = _words(transcript)
    return len(gt_words & got_words) / len(gt_words)


def _shared_stt_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{SHARED_STT_BASE_URL}/health", timeout=2)
        return True
    except Exception:  # noqa: BLE001
        return False


def _vllm_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{VLLM_BASE_URL}/models", timeout=2)
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# Pipeline stages - each returns (elapsed_seconds, text, extra: dict)
# --------------------------------------------------------------------------
def _whisper_transcribe(wav_bytes: bytes) -> tuple[float, str, dict]:

    boundary = "----benchmarkboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav_bytes + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ndistil-large-v3\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{SHARED_STT_BASE_URL}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
    elapsed = time.monotonic() - t0
    return elapsed, result.get("text", "").strip(), {}


def _gemma_text_reply(transcript: str) -> tuple[float, str, dict]:
    payload = {
        "model": LLM_MODEL,
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": RECEPTIONIST_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
    }
    req = urllib.request.Request(
        f"{VLLM_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    elapsed = time.monotonic() - t0
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return elapsed, text, {"prompt_tokens": usage.get("prompt_tokens")}


async def _direct_audio_transcribe(wav_bytes: bytes) -> tuple[float, str, dict]:
    async with gac.GemmaAudioClient(base_url=VLLM_BASE_URL, model=LLM_MODEL) as client:
        result = await client.transcribe_audio(wav_bytes)
    return result.latency_secs, result.text, {"prompt_tokens": result.prompt_tokens}


async def _direct_audio_single_call(wav_bytes: bytes) -> tuple[float, str, dict]:
    async with gac.GemmaAudioClient(base_url=VLLM_BASE_URL, model=LLM_MODEL) as client:
        result = await client.respond_to_audio(
            wav_bytes, system_prompt=RECEPTIONIST_SYSTEM_PROMPT
        )
    return result.latency_secs, result.text, {"prompt_tokens": result.prompt_tokens}


# --------------------------------------------------------------------------
def run_for_utterance(utt: dict) -> dict:
    import asyncio

    wav_bytes = utt["wav"].read_bytes()
    ground_truth = utt["ground_truth"]
    out: dict = {"utterance": utt["name"], "ground_truth": ground_truth}

    # --- 1. cascade: shared Whisper STT + separate Gemma text call ---
    if _shared_stt_reachable() and _vllm_reachable():
        stt_samples, llm_samples, total_samples = [], [], []
        last_transcript = last_reply = ""
        for _ in range(N_RUNS + 1):
            stt_s, transcript, _ = _whisper_transcribe(wav_bytes)
            llm_s, reply, _ = _gemma_text_reply(transcript)
            stt_samples.append(stt_s)
            llm_samples.append(llm_s)
            total_samples.append(stt_s + llm_s)
            last_transcript, last_reply = transcript, reply
        out["cascade"] = {
            "stt_stage": _summary(stt_samples[1:]),
            "llm_stage": _summary(llm_samples[1:]),
            "total": _summary(total_samples[1:]),
            "transcript": last_transcript,
            "reply": last_reply,
            "word_overlap": round(_word_overlap_ratio(ground_truth, last_transcript), 2),
        }
    else:
        out["cascade"] = {"skipped": "shared STT or vLLM not reachable"}

    # --- 2. direct_audio_stt: Gemma-as-STT + separate Gemma text call ---
    if _vllm_reachable():
        async def _run_direct_stt():
            stt_samples, llm_samples, total_samples = [], [], []
            last_transcript = last_reply = ""
            for _ in range(N_RUNS + 1):
                stt_s, transcript, _ = await _direct_audio_transcribe(wav_bytes)
                llm_s, reply, _ = _gemma_text_reply(transcript)
                stt_samples.append(stt_s)
                llm_samples.append(llm_s)
                total_samples.append(stt_s + llm_s)
                last_transcript, last_reply = transcript, reply
            return stt_samples[1:], llm_samples[1:], total_samples[1:], last_transcript, last_reply

        stt_s, llm_s, total_s, last_transcript, last_reply = asyncio.run(_run_direct_stt())
        out["direct_audio_stt"] = {
            "stt_stage": _summary(stt_s),
            "llm_stage": _summary(llm_s),
            "total": _summary(total_s),
            "transcript": last_transcript,
            "reply": last_reply,
            "word_overlap": round(_word_overlap_ratio(ground_truth, last_transcript), 2),
        }

        # --- 3. direct_audio_single_call: ONE call, audio -> reply ---
        async def _run_single_call():
            samples = []
            last_reply = ""
            for _ in range(N_RUNS + 1):
                s, reply, _ = await _direct_audio_single_call(wav_bytes)
                samples.append(s)
                last_reply = reply
            return samples[1:], last_reply

        samples, last_reply = asyncio.run(_run_single_call())
        out["direct_audio_single_call"] = {
            "total": _summary(samples),
            "reply": last_reply,
        }
    else:
        out["direct_audio_stt"] = {"skipped": "vLLM not reachable"}
        out["direct_audio_single_call"] = {"skipped": "vLLM not reachable"}

    return out


def main() -> None:
    print(
        f"vLLM: {VLLM_BASE_URL} ({'reachable' if _vllm_reachable() else 'NOT REACHABLE'}), "
        f"shared STT: {SHARED_STT_BASE_URL} "
        f"({'reachable' if _shared_stt_reachable() else 'NOT REACHABLE'})",
        file=sys.stderr,
    )

    results = []
    for utt in UTTERANCES:
        if not utt["wav"].exists():
            print(f"skipping {utt['name']}: fixture missing at {utt['wav']}", file=sys.stderr)
            continue
        print(f"--- {utt['name']} ---", file=sys.stderr)
        result = run_for_utterance(utt)
        results.append(result)
        print(json.dumps(result, indent=2), file=sys.stderr)

    out_path = THIS_DIR / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Quick human-readable rollup across all utterances that ran on both
    # cascade and direct_audio_stt (an apples-to-apples pipeline swap).
    both = [
        r
        for r in results
        if "total" in r.get("cascade", {}) and "total" in r.get("direct_audio_stt", {})
    ]
    if both:
        cascade_means = [r["cascade"]["total"]["mean_ms"] for r in both]
        direct_means = [r["direct_audio_stt"]["total"]["mean_ms"] for r in both]
        print(
            f"\nAcross {len(both)} utterances - "
            f"cascade avg total: {round(statistics.mean(cascade_means), 1)}ms, "
            f"direct_audio_stt avg total: {round(statistics.mean(direct_means), 1)}ms"
        )
        overlaps = [r["direct_audio_stt"]["word_overlap"] for r in both]
        print(f"direct_audio_stt transcript word-overlap vs ground truth: {overlaps}")


if __name__ == "__main__":
    main()
