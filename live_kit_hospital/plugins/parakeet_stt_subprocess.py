"""
NVIDIA Parakeet TDT as a LiveKit STT plugin - SUBPROCESS EDITION.

WHY A SUBPROCESS: nemo_toolkit[asr] currently fails to build on Python 3.12
(an old numba/llvmlite pin deep in librosa's dependency chain has no wheel
for 3.12 and won't build from source). Rather than fight that in the main
agent's venv, Parakeet runs in its OWN Python 3.10/3.11 venv
(see plugins/parakeet_worker.py) as an isolated subprocess, communicating
over stdin/stdout JSON - the exact same architecture as
plugins/qwen_tts.py's QwenSubprocessTTS, and for the same reason: total
dependency isolation between the two Python environments.

FIXES PORTED 1:1 FROM qwen_tts.py (same failure modes apply to any
subprocess worker on this project):
  - start_new_session=True: Ctrl+C sends SIGINT to the whole foreground
    process group; without this the worker dies uncontrolled instead of
    being torn down deterministically by _cleanup_process().
  - WORKER_READ_TIMEOUT_SECS: a wedged worker must error and get killed,
    not silently mute STT forever.
  - Per-request id tagging + stale-line discarding at DEBUG.
  - Respawn-on-crash via _process_alive() checks before each call.

ENV VARS (see .env.example):
  PARAKEET_PYTHON - path to the OTHER venv's python, e.g.
      /home/nauyan/voice-agent-pipeline/.venv-parakeet/bin/python
  PARAKEET_WORKER - path to parakeet_worker.py

If these aren't set, agent.py falls back to the in-process
plugins/parakeet_stt.py (requires nemo importable in the MAIN venv - will
fail on Python 3.12 per the README), and finally to faster-whisper. See
agent.py's _make_stt().
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid

import numpy as np
from loguru import logger

from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    stt,
    utils,
)

PARAKEET_SAMPLE_RATE = 16000
WORKER_READ_TIMEOUT_SECS = 30.0
WORKER_INIT_TIMEOUT_SECS = 180.0  # first run downloads ~2.4 GB + loads onto GPU


class ParakeetSubprocessSTT(stt.STT):
    def __init__(
        self,
        *,
        python_exec: str | None = None,
        worker_script: str | None = None,
        language: str = "en",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._python_exec = python_exec or os.environ["PARAKEET_PYTHON"]
        self._worker_script = worker_script or os.environ["PARAKEET_WORKER"]
        self._language = language
        self._process: asyncio.subprocess.Process | None = None
        self._spawn_lock = asyncio.Lock()
        # One transcription in flight at a time - mirrors the single-lane
        # executor in the in-process plugin, no GPU contention spikes
        # against vLLM mid-turn.
        self._call_lock = asyncio.Lock()

    # ------------------------------------------------------------ lifecycle
    def load(self) -> None:
        """Called synchronously from prewarm(proc) in agent.py. Spawning a
        subprocess needs a running event loop, so this just runs the async
        spawn+init to completion before prewarm returns."""
        asyncio.run(self._ensure_started())

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def _ensure_started(self) -> None:
        if self._process_alive():
            return
        async with self._spawn_lock:
            if self._process_alive():
                return
            logger.info(
                f"ParakeetSubprocessSTT: spawning worker via {self._python_exec} ..."
            )
            self._process = await asyncio.create_subprocess_exec(
                self._python_exec,
                self._worker_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,  # inherit - worker's load progress/errors show
                # up directly in the main agent's logs
                start_new_session=True,  # Ctrl+C isolation - see module docstring
            )
            self._process.stdin.write((json.dumps({"action": "init"}) + "\n").encode())
            await self._process.stdin.drain()
            try:
                ready_line = await asyncio.wait_for(
                    self._process.stdout.readline(), timeout=WORKER_INIT_TIMEOUT_SECS
                )
            except asyncio.TimeoutError:
                self._process.kill()
                self._process = None
                raise RuntimeError(
                    f"Parakeet worker did not report ready within "
                    f"{WORKER_INIT_TIMEOUT_SECS:.0f}s - run parakeet_worker.py "
                    f"standalone in its venv to debug (see its docstring)."
                )
            if not ready_line:
                self._process = None
                raise RuntimeError(
                    "Parakeet worker exited before reporting ready - check "
                    "stderr above (likely a nemo/torch import error in that venv)."
                )
            payload = json.loads(ready_line.decode("utf-8"))
            if not payload.get("ready"):
                self._process = None
                raise RuntimeError(
                    f"Parakeet worker init failed: {payload.get('error')}"
                )
            logger.info("ParakeetSubprocessSTT: worker ready (warm).")

    async def _cleanup_process(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.kill()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning(
                    "ParakeetSubprocessSTT: worker did not exit after kill()"
                )
        self._process = None

    # ------------------------------------------------------------- transcribe
    @staticmethod
    def _buffer_to_float32(buffer: utils.AudioBuffer) -> np.ndarray:
        frame = rtc.combine_audio_frames(buffer)
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        if frame.sample_rate != PARAKEET_SAMPLE_RATE:
            src_len = audio.shape[0]
            dst_len = int(round(src_len * PARAKEET_SAMPLE_RATE / frame.sample_rate))
            audio = np.interp(
                np.linspace(0.0, src_len - 1, dst_len, dtype=np.float64),
                np.arange(src_len, dtype=np.float64),
                audio,
            ).astype(np.float32)
        return audio

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        await self._ensure_started()
        audio = self._buffer_to_float32(buffer)
        audio_b64 = base64.b64encode(audio.tobytes()).decode()
        req_id = uuid.uuid4().hex

        async with self._call_lock:
            cmd = {
                "action": "transcribe",
                "id": req_id,
                "audio_b64": audio_b64,
                "sample_rate": PARAKEET_SAMPLE_RATE,
            }
            try:
                self._process.stdin.write((json.dumps(cmd) + "\n").encode())
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._process = None  # force respawn on the next call
                raise RuntimeError(f"Parakeet worker pipe broken: {exc}") from exc

            while True:
                try:
                    line = await asyncio.wait_for(
                        self._process.stdout.readline(),
                        timeout=WORKER_READ_TIMEOUT_SECS,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"ParakeetSubprocessSTT: no output for "
                        f"{WORKER_READ_TIMEOUT_SECS}s (id={req_id}), killing worker"
                    )
                    await self._cleanup_process()
                    raise RuntimeError("Parakeet worker timed out")

                if not line:
                    self._process = None
                    raise RuntimeError("Parakeet worker exited unexpectedly")

                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                if data.get("id") != req_id:
                    logger.debug(
                        f"ParakeetSubprocessSTT: discarding stale line for "
                        f"id={data.get('id')} (current id={req_id})"
                    )
                    continue

                if "error" in data:
                    raise RuntimeError(f"Parakeet worker error: {data['error']}")

                text = data.get("text", "")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(text=text, language=language or self._language)
                    ],
                )
