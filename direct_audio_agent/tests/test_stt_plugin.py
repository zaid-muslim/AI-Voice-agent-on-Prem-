"""
Tests for stt_plugin.py's GemmaDirectAudioSTT.

Unit tests verify the audio-buffer-to-WAV plumbing and that a failed
direct-audio call surfaces as the livekit-agents-standard
APIConnectionError (so the framework's own retry/error handling applies
same as any other STT plugin), using a monkeypatched transcribe_audio so
no network is involved.

The live test drives the REAL public STT interface (`stt.recognize()`,
the same method AgentSession itself calls) with real rtc.AudioFrames
built from the same known-sentence WAV fixture used in
test_gemma_audio_client.py - this is the "does the plugin actually work
end-to-end, not just in isolation" check. Skips if vLLM is unreachable.

Run:
    pytest direct_audio_agent/tests/test_stt_plugin.py -v
"""

from __future__ import annotations

import asyncio
import socket
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemma_audio_client as gac  # noqa: E402
from stt_plugin import GemmaDirectAudioSTT  # noqa: E402

from livekit import rtc  # noqa: E402
from livekit.agents import APIConnectionError, stt  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_WAV = FIXTURES / "sample_cardiology_utterance.wav"
SAMPLE_WAV_EXPECTED_WORDS = ("cardiology", "appointment", "tomorrow")


def _vllm_reachable(host: str = "127.0.0.1", port: int = 8000, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _frame(pcm: np.ndarray, sample_rate: int, num_channels: int = 1) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=pcm.tobytes(),
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=pcm.shape[0] // num_channels,
    )


def _wav_file_to_frames(path: Path, chunk_ms: int = 100) -> list[rtc.AudioFrame]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    chunk_len = int(rate * chunk_ms / 1000) * channels
    return [
        _frame(pcm[i : i + chunk_len], rate, channels)
        for i in range(0, len(pcm), chunk_len)
        if len(pcm[i : i + chunk_len]) > 0
    ]


# --------------------------------------------------------------------------
# Buffer -> PCM plumbing (no network)
# --------------------------------------------------------------------------
def test_buffer_to_pcm16_mono_16k_resamples_and_downmixes():
    # 0.5s of stereo audio @ 22050Hz -> should come out mono @ 16000Hz.
    stereo = np.zeros(int(22_050 * 0.5) * 2, dtype=np.int16)
    frame = _frame(stereo, sample_rate=22_050, num_channels=2)

    pcm16k = GemmaDirectAudioSTT._buffer_to_pcm16_mono_16k([frame])

    assert pcm16k.dtype == np.int16
    assert pcm16k.shape[0] == pytest.approx(8_000, abs=2)  # 0.5s @ 16kHz


def test_buffer_to_pcm16_mono_16k_from_real_fixture_matches_duration():
    if not SAMPLE_WAV.exists():
        pytest.skip(f"fixture missing: {SAMPLE_WAV}")
    frames = _wav_file_to_frames(SAMPLE_WAV)
    with wave.open(str(SAMPLE_WAV), "rb") as w:
        expected_secs = w.getnframes() / w.getframerate()

    pcm16k = GemmaDirectAudioSTT._buffer_to_pcm16_mono_16k(frames)

    assert pcm16k.shape[0] / gac.SAMPLE_RATE == pytest.approx(expected_secs, abs=0.05)


# --------------------------------------------------------------------------
# _recognize_impl error handling (monkeypatched client, no network)
# --------------------------------------------------------------------------
def test_recognize_impl_wraps_gemma_audio_error_as_api_connection_error(monkeypatch):
    async def _boom(self, wav_bytes, **kwargs):
        raise gac.GemmaAudioError("server unreachable (simulated)")

    monkeypatch.setattr(gac.GemmaAudioClient, "transcribe_audio", _boom)

    async def _run():
        from livekit.agents import DEFAULT_API_CONNECT_OPTIONS

        sut = GemmaDirectAudioSTT()
        silence = np.zeros(8_000, dtype=np.int16)
        frame = _frame(silence, sample_rate=16_000)
        try:
            await sut._recognize_impl(
                [frame], language=None, conn_options=DEFAULT_API_CONNECT_OPTIONS
            )
        finally:
            await sut.aclose()

    with pytest.raises(APIConnectionError):
        asyncio.run(_run())


def test_recognize_impl_returns_final_transcript_event(monkeypatch):
    async def _fake_transcribe(self, wav_bytes, **kwargs):
        return gac.AudioTurnResult(text="hello from gemma", latency_secs=0.1, prompt_tokens=10)

    monkeypatch.setattr(gac.GemmaAudioClient, "transcribe_audio", _fake_transcribe)

    async def _run():
        sut = GemmaDirectAudioSTT(language="en")
        silence = np.zeros(8_000, dtype=np.int16)
        frame = _frame(silence, sample_rate=16_000)
        try:
            from livekit.agents import DEFAULT_API_CONNECT_OPTIONS

            return await sut._recognize_impl(
                [frame], language=None, conn_options=DEFAULT_API_CONNECT_OPTIONS
            )
        finally:
            await sut.aclose()

    event = asyncio.run(_run())
    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "hello from gemma"
    assert event.alternatives[0].language == "en"


# --------------------------------------------------------------------------
# Live: the real public stt.recognize() path, real fixture audio
# --------------------------------------------------------------------------
def test_live_recognize_matches_known_sentence():
    if not _vllm_reachable():
        pytest.skip("vLLM not reachable at 127.0.0.1:8000 - skipping live check")
    if not SAMPLE_WAV.exists():
        pytest.skip(f"fixture missing: {SAMPLE_WAV}")

    async def _run():
        sut = GemmaDirectAudioSTT()
        frames = _wav_file_to_frames(SAMPLE_WAV)
        try:
            return await sut.recognize(frames)
        finally:
            await sut.aclose()

    event = asyncio.run(_run())
    text = event.alternatives[0].text.lower()
    missing = [w for w in SAMPLE_WAV_EXPECTED_WORDS if w not in text]
    assert not missing, f"transcript missing expected words {missing}: got {text!r}"
