"""
Real end-to-end latency test: join the ACTUAL running LiveKit room as a
synthetic caller, publish a real recorded utterance in real time, and
measure wall-clock time from "caller stops talking" to "agent's response
audio starts" - through the FULL real pipeline (VAD, turn detector, STT,
LLM, TTS - not a composed estimate from tests/manual/benchmark_latency.py's
independent per-stage numbers, which cannot capture endpointing wait,
tool-call round trips, or real reply-length effects on TTS).

WHY THIS EXISTS: benchmark_latency.py's composed numbers (~450-700ms)
significantly UNDERSTATED real conversational latency. Measured live
against this project's own stack: **~3.2-3.5 seconds**, stop-talking to
agent-audio-starts, for a real booking/informational request. The gap is
real, not a measurement artifact - see README §4 for the full breakdown
and why (mainly: endpointing/turn-detection wait, and most real requests
trigger a tool call, meaning TWO LLM round trips - decide to call the
tool, then respond to its result - not one).

TWO NON-OBVIOUS BUGS THIS SCRIPT HAD TO WORK AROUND, keep both fixes if
you modify this:
1. A published WebRTC audio track streams CONTINUOUSLY once active
   (silence included) - "did a new frame arrive" is NOT a valid "is the
   agent currently speaking" signal. This script tracks actual per-frame
   audio ENERGY (max abs sample value vs SILENCE_THRESHOLD), not frame
   arrival, to detect real speech vs real silence.
2. The caller's audio track MUST be published with
   TrackPublishOptions(source=SOURCE_MICROPHONE) - LiveKit Agents'
   RoomIO only attaches to that source. Without it, the agent never
   processes the caller's audio at all (confirmed live: no VAD, no STT,
   no reply - the log shows "input stream attached" with
   source=SOURCE_UNKNOWN and nothing else ever happens for that track).

SETUP (all bare-metal, one terminal each - or point env vars at whatever
combination of these is already running):
    livekit-server --config app/livekit.yaml --dev   # with LIVEKIT_KEYS set
    bash scripts/run_vllm.sh
    python stt_service/server.py
    cd app && python main.py dev
Then generate a caller utterance once (any of this project's own TTS
engines works - this reuses the real production plugin so the WAV sounds
like real, clean speech, not synthetic noise):
    python -c "
import asyncio, sys, wave
sys.path.insert(0, 'app')
from plugins.shared_qwen_tts import SharedQwenTTS
async def main():
    tts = SharedQwenTTS()
    stream = tts.synthesize('Hi, I need to book an appointment with cardiology for next Tuesday morning.')
    frames, sr = [], None
    async for ev in stream:
        frames.append(bytes(ev.frame.data)); sr = ev.frame.sample_rate
    with wave.open('caller_utterance.wav', 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(b''.join(frames))
    await tts.aclose()
asyncio.run(main())
"
Then run this (unbuffered output matters - progress prints won't show
otherwise when not attached to a TTY):
    LIVEKIT_URL=ws://localhost:7880 \\
    LIVEKIT_API_KEY=<key> LIVEKIT_API_SECRET=<secret> \\
    E2E_WAV_PATH=./caller_utterance.wav \\
    python -u tests/manual/e2e_call_latency_test.py
"""

from __future__ import annotations

import asyncio
import functools
import os
import struct
import time
import uuid
import wave

from livekit import api, rtc

print = functools.partial(print, flush=True)  # noqa: A001 - unbuffered, this runs non-interactively

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
API_KEY = os.environ["LIVEKIT_API_KEY"]
API_SECRET = os.environ["LIVEKIT_API_SECRET"]
WAV_PATH = os.environ.get("E2E_WAV_PATH", "caller_utterance.wav")

ROOM_NAME = f"e2e-test-{uuid.uuid4().hex[:8]}"
FRAME_MS = 20
SILENCE_THRESHOLD = 400  # out of 32767 - well above digital-silence noise floor
GREETING_QUIET_SECS = 1.0  # how long the agent must be quiet before we consider its greeting done
RESPONSE_TIMEOUT_SECS = 20


def _load_wav(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sr


def _frame_max_abs(data: bytes) -> int:
    if not data:
        return 0
    n = len(data) // 2
    samples = struct.unpack(f"<{n}h", data[: n * 2])
    return max(abs(s) for s in samples) if samples else 0


async def main() -> None:
    pcm, sr = _load_wav(WAV_PATH)
    samples_per_frame = int(sr * FRAME_MS / 1000)
    bytes_per_frame = samples_per_frame * 2
    print(f"Loaded caller utterance: {len(pcm) / 2 / sr:.2f}s @ {sr}Hz")

    token = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity("e2e-caller")
        .with_name("E2E Test Caller")
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME))
        .to_jwt()
    )

    room = rtc.Room()
    agent_events: list[tuple[float, bool]] = []  # (timestamp, is_speech)
    agent_ready = asyncio.Event()

    @room.on("track_subscribed")
    def _on_track_subscribed(track, publication, participant):  # noqa: ANN001
        if track.kind == rtc.TrackKind.KIND_AUDIO and participant.identity != "e2e-caller":
            print(f"[{time.perf_counter():.3f}] Subscribed to agent audio track ({participant.identity})")
            stream = rtc.AudioStream(track)

            async def _consume():
                async for event in stream:
                    amp = _frame_max_abs(bytes(event.frame.data))
                    agent_events.append((time.perf_counter(), amp > SILENCE_THRESHOLD))
                    agent_ready.set()

            asyncio.ensure_future(_consume())

    print(f"Connecting to room {ROOM_NAME} ...")
    await room.connect(LIVEKIT_URL, token)
    print(f"[{time.perf_counter():.3f}] Connected - agent should auto-dispatch into this room.")

    source = rtc.AudioSource(sr, 1)
    track = rtc.LocalAudioTrack.create_audio_track("caller-mic", source)
    # source=SOURCE_MICROPHONE is REQUIRED - see module docstring bug #2.
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    print(f"[{time.perf_counter():.3f}] Published caller audio track (source=MICROPHONE).")

    silence_frame_bytes = bytes(bytes_per_frame)
    stop_silence = asyncio.Event()

    async def silence_feeder():
        while not stop_silence.is_set():
            frame = rtc.AudioFrame(
                data=silence_frame_bytes, sample_rate=sr, num_channels=1, samples_per_channel=samples_per_frame
            )
            await source.capture_frame(frame)

    feeder_task = asyncio.ensure_future(silence_feeder())

    print("Waiting for agent to join and speak its greeting...")
    try:
        await asyncio.wait_for(agent_ready.wait(), timeout=30)
    except asyncio.TimeoutError:
        print("ERROR: agent never published audio (never joined, or never spoke).")
        stop_silence.set()
        await room.disconnect()
        return

    while not any(is_speech for _, is_speech in agent_events):
        await asyncio.sleep(0.05)
    print(f"[{time.perf_counter():.3f}] Greeting speech actually started.")

    last_speech_time = None
    while True:
        await asyncio.sleep(0.1)
        speech_times = [t for t, is_speech in agent_events if is_speech]
        if speech_times:
            last_speech_time = speech_times[-1]
        if last_speech_time is not None and (time.perf_counter() - last_speech_time) >= GREETING_QUIET_SECS:
            print(f"[{time.perf_counter():.3f}] Greeting finished ({len(agent_events)} frames so far).")
            break

    # --- speak the real utterance in real time ---
    stop_silence.set()
    await feeder_task
    print(f"[{time.perf_counter():.3f}] Speaking real utterance now...")
    n_frames = len(pcm) // bytes_per_frame
    for i in range(n_frames):
        chunk = pcm[i * bytes_per_frame : (i + 1) * bytes_per_frame]
        frame = rtc.AudioFrame(data=chunk, sample_rate=sr, num_channels=1, samples_per_channel=samples_per_frame)
        await source.capture_frame(frame)
    t_stop_speaking = time.perf_counter()
    print(f"[{t_stop_speaking:.3f}] Finished speaking utterance (t_stop_speaking).")

    # --- resume silence, wait for the agent's response speech ---
    stop_silence.clear()
    feeder_task = asyncio.ensure_future(silence_feeder())
    print("Waiting for agent response speech energy...")

    # Anchor on the EVENT INDEX at this exact moment, not a timestamp
    # comparison - confirmed live this matters: a stray/unrelated prior
    # exchange's trailing audio can still be arriving (network jitter
    # buffer) right around t_stop_speaking, and comparing by wall-clock
    # timestamp alone can misattribute that tail as "the response,"
    # ending the test before the real reply is even generated.
    frames_before_response = len(agent_events)
    deadline = time.perf_counter() + RESPONSE_TIMEOUT_SECS
    t_first_response = None
    checked_idx = frames_before_response
    while time.perf_counter() < deadline:
        await asyncio.sleep(0.02)
        while checked_idx < len(agent_events):
            t, is_speech = agent_events[checked_idx]
            checked_idx += 1
            if is_speech:
                t_first_response = t
                break
        if t_first_response is not None:
            break

    stop_silence.set()
    await feeder_task

    if t_first_response is None:
        print(f"ERROR: no agent response speech energy detected within {RESPONSE_TIMEOUT_SECS}s")
    else:
        latency = t_first_response - t_stop_speaking
        print("\n=== RESULT ===")
        print(f"END-TO-END LATENCY (stop talking -> agent audio starts): {latency * 1000:.1f} ms")

    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
