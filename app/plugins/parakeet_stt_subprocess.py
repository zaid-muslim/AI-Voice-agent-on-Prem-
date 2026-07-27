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
import threading
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
        self._stderr_task: asyncio.Task | None = None
        self._spawn_lock = asyncio.Lock()
        # One transcription in flight at a time - mirrors the single-lane
        # executor in the in-process plugin, no GPU contention spikes
        # against vLLM mid-turn.
        self._call_lock = asyncio.Lock()

        # DEDICATED BACKGROUND EVENT LOOP - this is the actual fix for a
        # real bug: load() is called synchronously from prewarm(proc),
        # which has NO running event loop yet. The tempting fix is
        # asyncio.run(self._ensure_started()) - but asyncio.run() creates
        # a throwaway loop, runs the coroutine, then DESTROYS that loop
        # when done. The subprocess transport it creates stays alive as an
        # object, but is permanently bound to that now-dead loop. The
        # first real call later - on LiveKit's actual session event loop,
        # a totally different loop - then fails with "Event loop is
        # closed" the moment it tries to write to that orphaned
        # transport. The fix: run ONE persistent loop, forever, in its own
        # background thread, created once and never torn down for the
        # life of this object. Every operation on self._process - spawn,
        # write, read, cleanup - is dispatched onto THIS loop via
        # run_coroutine_threadsafe, regardless of which loop the caller
        # (prewarm's throwaway context, or the real session loop) happens
        # to be running on.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_start_lock = threading.Lock()  # plain thread lock -
        # _ensure_loop() may be called before any event loop exists at all

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start the dedicated background loop/thread once, idempotently.
        Safe to call from prewarm's sync context or from async code on any
        other loop - this never touches the CALLER's loop, only ever
        creates/returns our own private one."""
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
                name="parakeet-subprocess-loop",
            )
            self._loop_thread.start()
            return self._loop

    # ------------------------------------------------------------ lifecycle
    def load(self) -> None:
        """Called synchronously from prewarm(proc) in agent.py - no event
        loop is running yet at this point. Ensure our persistent
        background loop exists, then submit the real async spawn+init work
        to it and block (via .result()) until that finishes, so prewarm()
        still doesn't return until Parakeet is genuinely warm."""
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
                f"ParakeetSubprocessSTT: spawning worker via {self._python_exec} ..."
            )
            self._process = await asyncio.create_subprocess_exec(
                self._python_exec,
                self._worker_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                # NEVER inherit stderr here (stderr=None used to mean
                # exactly that). The LiveKit job process that calls this
                # is itself using ITS OWN stdio as an internal IPC channel
                # back to the main worker process - if our subprocess
                # shares that same fd, NeMo's chatty internal logging
                # (which itself throws BrokenPipeErrors under load) writes
                # directly into LiveKit's own message stream and corrupts
                # it, surfacing as a confusing "Expecting value" JSON
                # decode error on LiveKit's side, nowhere near the real
                # cause. A fully separate pipe, drained by us below,
                # isolates this subprocess's IO completely.
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # Ctrl+C isolation - see module docstring
            )
            self._stderr_task = asyncio.create_task(self._drain_stderr())
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

    async def _drain_stderr(self) -> None:
        """Continuously read the worker's stderr (NeMo's load progress,
        warnings, tracebacks) and forward it through OUR OWN logger, on our
        own separately-piped fd - never the parent's inherited stream. This
        is what makes it safe to pipe stderr instead of inheriting it: we
        still see everything the worker prints, just without any risk of
        it colliding with LiveKit's own internal IPC."""
        if self._process is None or self._process.stderr is None:
            return
        try:
            async for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug(f"[parakeet_worker stderr] {text}")
        except (asyncio.CancelledError, ValueError):
            pass  # normal on shutdown/respawn

    async def _cleanup_process(self) -> None:
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None
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
        """Runs on WHATEVER loop LiveKit's session is using. Don't touch
        self._process directly here - dispatch the real work onto our
        dedicated background loop (same one load() used), and bridge its
        result back to this loop with asyncio.wrap_future(). This is what
        actually fixes the "Event loop is closed" crash: self._process is
        now only ever read/written from the ONE loop it was created on."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._do_recognize(buffer, language), loop
        )
        return await asyncio.wrap_future(future)

    async def _do_recognize(
        self, buffer: utils.AudioBuffer, language: str | None
    ) -> stt.SpeechEvent:
        """The actual transcription request/response - always executed on
        our dedicated background loop (see _recognize_impl above), so
        self._process is always accessed from the same loop it was
        created on."""
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
