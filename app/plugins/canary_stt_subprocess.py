"""
NVIDIA Canary as a LiveKit STT plugin - subprocess edition.

Identical architecture to plugins/parakeet_stt_subprocess.py, including
the DEDICATED BACKGROUND EVENT LOOP fix (see that file's docstring for
the full "Event loop is closed" story from earlier this week) - this
plugin has the exact same risk (warmed via prewarm(proc)'s synchronous,
no-running-loop context), so it needs the exact same fix. Rather than
duplicate that long explanation here, the short version: never let
asyncio.run() create a throwaway loop for a subprocess meant to outlive
that call - run one persistent loop, forever, in a background thread.

ENV VARS:
  CANARY_PYTHON - path to the venv's python (same venv as Parakeet works
      for this, per canary_worker.py's docstring)
  CANARY_WORKER - path to canary_worker.py
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid

from loguru import logger

from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    stt,
    utils,
)

CANARY_SAMPLE_RATE = 16000
WORKER_READ_TIMEOUT_SECS = 30.0
WORKER_INIT_TIMEOUT_SECS = 300.0


class CanarySubprocessSTT(stt.STT):
    def __init__(
        self,
        *,
        python_exec: str | None = None,
        worker_script: str | None = None,
        model: str = "nvidia/canary-180m-flash",
        language: str = "en",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._python_exec = python_exec or os.environ["CANARY_PYTHON"]
        self._worker_script = worker_script or os.environ["CANARY_WORKER"]
        self._model = model
        self._language = language
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None
        self._spawn_lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()

        # Dedicated background loop - see module docstring / the original
        # Parakeet subprocess fix this week for why this is load-bearing,
        # not optional.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_start_lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        with self._loop_start_lock:
            if self._loop is not None:
                return self._loop
            self._loop = asyncio.new_event_loop()

            def _run_forever(loop: asyncio.AbstractEventLoop) -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            self._loop_thread = threading.Thread(
                target=_run_forever,
                args=(self._loop,),
                daemon=True,
                name="canary-subprocess-loop",
            )
            self._loop_thread.start()
            return self._loop

    def load(self) -> None:
        """Called synchronously from prewarm(proc)."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._ensure_started(), loop)
        future.result(timeout=WORKER_INIT_TIMEOUT_SECS + 30)

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def _ensure_started(self) -> None:
        if self._process_alive():
            return
        async with self._spawn_lock:
            if self._process_alive():
                return
            logger.info(
                f"CanarySubprocessSTT: spawning worker via {self._python_exec} ..."
            )
            self._process = await asyncio.create_subprocess_exec(
                self._python_exec,
                self._worker_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,  # never inherit - see
                # parakeet_stt_subprocess.py for why this is load-bearing
                start_new_session=True,
            )
            self._stderr_task = asyncio.create_task(self._drain_stderr())

            init_cmd = {"action": "init", "model": self._model}
            self._process.stdin.write((json.dumps(init_cmd) + "\n").encode())
            await self._process.stdin.drain()
            try:
                ready_line = await asyncio.wait_for(
                    self._process.stdout.readline(), timeout=WORKER_INIT_TIMEOUT_SECS
                )
            except asyncio.TimeoutError:
                self._process.kill()
                self._process = None
                raise RuntimeError(
                    f"Canary worker did not report ready within "
                    f"{WORKER_INIT_TIMEOUT_SECS:.0f}s - run canary_worker.py "
                    f"standalone in its venv to debug."
                )
            if not ready_line:
                self._process = None
                raise RuntimeError(
                    "Canary worker exited before reporting ready - check stderr above."
                )
            payload = json.loads(ready_line.decode("utf-8"))
            if not payload.get("ready"):
                self._process = None
                raise RuntimeError(f"Canary worker init failed: {payload.get('error')}")
            logger.info("CanarySubprocessSTT: worker ready (warm).")

    async def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            async for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug(f"[canary_worker stderr] {text}")
        except (asyncio.CancelledError, ValueError):
            pass

    async def _cleanup_process(self) -> None:
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None
        if self._process and self._process.returncode is None:
            self._process.kill()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning("CanarySubprocessSTT: worker did not exit after kill()")
        self._process = None

    @staticmethod
    def _buffer_to_float32(buffer: utils.AudioBuffer) -> "object":
        import numpy as np

        frame = rtc.combine_audio_frames(buffer)
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        if frame.sample_rate != CANARY_SAMPLE_RATE:
            src_len = audio.shape[0]
            dst_len = int(round(src_len * CANARY_SAMPLE_RATE / frame.sample_rate))
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
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._do_recognize(buffer, language), loop
        )
        return await asyncio.wrap_future(future)

    async def _do_recognize(
        self, buffer: utils.AudioBuffer, language: str | None
    ) -> stt.SpeechEvent:
        import base64 as _b64

        await self._ensure_started()
        audio = self._buffer_to_float32(buffer)
        audio_b64 = _b64.b64encode(audio.tobytes()).decode()
        req_id = uuid.uuid4().hex

        async with self._call_lock:
            cmd = {
                "action": "transcribe",
                "id": req_id,
                "audio_b64": audio_b64,
                "sample_rate": CANARY_SAMPLE_RATE,
            }
            try:
                self._process.stdin.write((json.dumps(cmd) + "\n").encode())
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._process = None
                raise RuntimeError(f"Canary worker pipe broken: {exc}") from exc

            while True:
                try:
                    line = await asyncio.wait_for(
                        self._process.stdout.readline(),
                        timeout=WORKER_READ_TIMEOUT_SECS,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"CanarySubprocessSTT: no output for "
                        f"{WORKER_READ_TIMEOUT_SECS}s (id={req_id}), killing worker"
                    )
                    await self._cleanup_process()
                    raise RuntimeError("Canary worker timed out")

                if not line:
                    self._process = None
                    raise RuntimeError("Canary worker exited unexpectedly")

                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                if data.get("id") != req_id:
                    logger.debug(
                        f"CanarySubprocessSTT: discarding stale line for "
                        f"id={data.get('id')} (current id={req_id})"
                    )
                    continue

                if "error" in data:
                    raise RuntimeError(f"Canary worker error: {data['error']}")

                text = data.get("text", "")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(text=text, language=language or self._language)
                    ],
                )
