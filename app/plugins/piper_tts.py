"""
Piper TTS as a LiveKit TTS plugin - subprocess edition.

*** GPL-3.0 LICENSE CAVEAT - see plugins/piper_worker.py's docstring for
the full explanation. Real, not a formality - check it fits your
deployment before shipping this engine commercially. ***

Same subprocess architecture as the other *_tts.py plugins in this
project (warmed via entrypoint(), never prewarm(proc); stderr piped
never inherited). One real difference from Chatterbox/Kokoro/Qwen:
Piper's sample rate is PER-VOICE (defined in that voice's config, not a
fixed constant for the whole engine) - this plugin reads it from the
worker's own reported sample_rate on the first response rather than
assuming a value, since assuming would be exactly the kind of
unverified-API mistake this project has been actively avoiding.

ENV VARS:
  PIPER_PYTHON - path to the dedicated venv's python (CPU-only is fine -
      Piper's own dependencies are onnxruntime + pathvalidate, no torch
      needed for inference)
  PIPER_WORKER - path to piper_worker.py
  PIPER_MODEL_PATH - path to the .onnx voice file (system_config.json's
      tts.model field for this engine - see piper_worker.py's docstring
      for where to download voices)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid

from loguru import logger

from livekit.agents import tts

DEFAULT_SAMPLE_RATE = 22050  # Piper's common default (medium-quality
# voices); OVERRIDDEN per-response from the worker's actual reported
# sample_rate once synthesis starts - this constant only matters for the
# brief window before the first real response arrives.
WORKER_READ_TIMEOUT_SECS = 30.0
WORKER_INIT_TIMEOUT_SECS = 60.0  # Piper loads fast (CPU, ONNX, no CUDA
# warm-up needed) - much shorter than the GPU engines' timeouts


class PiperSubprocessTTS(tts.TTS):
    def __init__(
        self,
        *,
        python_exec: str | None = None,
        worker_script: str | None = None,
        model_path: str | None = None,
        use_cuda: bool = False,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._python_exec = python_exec or os.environ["PIPER_PYTHON"]
        self._worker_script = worker_script or os.environ["PIPER_WORKER"]
        self._model_path = model_path or os.environ["PIPER_MODEL_PATH"]
        self._use_cuda = use_cuda
        self._process: asyncio.subprocess.Process | None = None
        self._spawn_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._stderr_task: asyncio.Task | None = None

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def prewarm(self) -> None:
        await self._ensure_started()

    async def _ensure_started(self) -> None:
        if self._process_alive():
            return
        async with self._spawn_lock:
            if self._process_alive():
                return
            logger.info(
                f"PiperSubprocessTTS: spawning worker via {self._python_exec} ..."
            )
            self._process = await asyncio.create_subprocess_exec(
                self._python_exec,
                self._worker_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            self._stderr_task = asyncio.create_task(self._drain_stderr())

            init_cmd = {
                "action": "init",
                "model_path": self._model_path,
                "use_cuda": self._use_cuda,
            }
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
                    f"Piper worker did not report ready within "
                    f"{WORKER_INIT_TIMEOUT_SECS:.0f}s - check that "
                    f"'{self._model_path}' and its matching .json exist."
                )
            if not ready_line:
                self._process = None
                raise RuntimeError(
                    "Piper worker exited before reporting ready - check stderr above."
                )
            payload = json.loads(ready_line.decode("utf-8"))
            if not payload.get("ready"):
                self._process = None
                raise RuntimeError(f"Piper worker init failed: {payload.get('error')}")
            logger.info("PiperSubprocessTTS: worker ready (warm).")

    async def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            async for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug(f"[piper_worker stderr] {text}")
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
                logger.warning("PiperSubprocessTTS: worker did not exit after kill()")
        self._process = None

    def synthesize(self, text: str, *, conn_options=None) -> "PiperChunkedStream":
        return PiperChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class PiperChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: PiperSubprocessTTS, input_text: str, conn_options=None):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter) -> None:
        await self._tts._ensure_started()
        proc = self._tts._process

        req_id = uuid.uuid4().hex
        emitter_initialized = False

        async with self._tts._request_lock:
            cmd = {"action": "tts", "id": req_id, "text": self.input_text}
            try:
                proc.stdin.write((json.dumps(cmd) + "\n").encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._tts._process = None
                raise RuntimeError(f"Piper worker pipe broken: {exc}") from exc

            while True:
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=WORKER_READ_TIMEOUT_SECS
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"PiperSubprocessTTS: no output for "
                        f"{WORKER_READ_TIMEOUT_SECS}s (id={req_id}), killing worker"
                    )
                    await self._tts._cleanup_process()
                    raise RuntimeError("Piper worker timed out")

                if not line:
                    self._tts._process = None
                    raise RuntimeError("Piper worker exited unexpectedly")

                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                if data.get("id") != req_id:
                    logger.debug(
                        f"PiperSubprocessTTS: discarding stale line for "
                        f"id={data.get('id')} (current id={req_id})"
                    )
                    continue

                if data.get("done"):
                    break
                if "error" in data:
                    raise RuntimeError(f"Piper worker error: {data['error']}")
                if "audio_b64" in data:
                    if not emitter_initialized:
                        # Piper's sample rate is PER-VOICE - read it from
                        # the worker's actual first response instead of
                        # assuming DEFAULT_SAMPLE_RATE is correct for
                        # whichever .onnx voice is actually loaded.
                        output_emitter.initialize(
                            request_id=req_id,
                            sample_rate=data.get("sample_rate", self._tts.sample_rate),
                            num_channels=1,
                            mime_type="audio/pcm",
                        )
                        emitter_initialized = True
                    output_emitter.push(base64.b64decode(data["audio_b64"]))

        if not emitter_initialized:
            # No audio at all (e.g. empty text) - still initialize so the
            # stream contract is honored, then immediately flush empty.
            output_emitter.initialize(
                request_id=req_id,
                sample_rate=self._tts.sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )
        output_emitter.flush()
