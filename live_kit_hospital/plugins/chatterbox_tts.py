"""
Chatterbox TTS (Resemble AI) as a LiveKit TTS plugin - subprocess edition.

WHY WARMED IN entrypoint(), NOT prewarm(proc) - THIS MATTERS:
Recall the real bug found this week in the Parakeet subprocess plugin:
calling asyncio.run() inside prewarm(proc) (a synchronous, no-running-loop
context) permanently ties the subprocess's transport to a throwaway event
loop that gets destroyed the moment prewarm() returns - causing "Event
loop is closed" the first time a REAL call tried to use it from the
actual session loop. The fix there was a dedicated persistent background
loop. This plugin sidesteps that whole class of bug by following
qwen_tts.py's ORIGINAL pattern instead: prewarm() here is a plain async
method, awaited directly inside agent.py's entrypoint() (which already
has a real, long-lived event loop running) - never invoked from
prewarm(proc)'s synchronous context at all. Simpler, and correct for the
same underlying reason.

HONEST LIMITATION (see chatterbox_worker.py's docstring for the full
explanation): Chatterbox's generate() call is NOT incremental - the
worker synthesizes the ENTIRE utterance before any audio chunk is sent.
This plugin's "first chunk" timing is therefore really "full synthesis
time," not a true streaming head start the way some TTS engines achieve.
For a hospital receptionist's typically-short responses this may still be
acceptable, but do NOT assume it behaves like Qwen's plugin under the
hood - measure it the same way the whisper/Parakeet STT comparison was
measured, don't assume.

NOT YET RUN END-TO-END (same caveat as chatterbox_worker.py): this
plugin's protocol-handling logic is verified against chatterbox_worker.py
running with a stub model. The real model has not been exercised with
real GPU + real weights in this build environment.

ENV VARS (mirrors the Parakeet subprocess pattern):
  CHATTERBOX_PYTHON - path to the dedicated venv's python, e.g.
      /home/nauyan/voice-agent-pipeline/.venv-chatterbox/bin/python
  CHATTERBOX_WORKER - path to chatterbox_worker.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid

from loguru import logger

from livekit.agents import tts
from livekit.agents.utils import AudioBuffer  # noqa: F401  (parity import; unused directly here)

CHATTERBOX_SAMPLE_RATE = 24000  # S3GEN_SR, confirmed from the real source
WORKER_READ_TIMEOUT_SECS = 30.0
WORKER_INIT_TIMEOUT_SECS = 300.0  # generous: covers a possible first-time
# Hugging Face model download on top of normal load time


class ChatterboxSubprocessTTS(tts.TTS):
    def __init__(
        self,
        *,
        python_exec: str | None = None,
        worker_script: str | None = None,
        sample_rate: int = CHATTERBOX_SAMPLE_RATE,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._python_exec = python_exec or os.environ["CHATTERBOX_PYTHON"]
        self._worker_script = worker_script or os.environ["CHATTERBOX_WORKER"]
        self._process: asyncio.subprocess.Process | None = None
        self._spawn_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()  # one synthesis in flight at a
        # time - Chatterbox's non-incremental generate() call is
        # presumably not safe/sensible to run concurrently against a
        # single loaded model instance in one worker process anyway.
        self._current_req_id: str | None = None
        self._stderr_task: asyncio.Task | None = None

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def prewarm(self) -> None:
        """Called directly with `await` from agent.py's entrypoint() - a
        REAL running event loop, not prewarm(proc)'s synchronous context.
        This is deliberate; see module docstring."""
        await self._ensure_started()

    async def _ensure_started(self) -> None:
        if self._process_alive():
            return
        async with self._spawn_lock:
            if self._process_alive():
                return
            logger.info(
                f"ChatterboxSubprocessTTS: spawning worker via {self._python_exec} ..."
            )
            self._process = await asyncio.create_subprocess_exec(
                self._python_exec,
                self._worker_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,  # NEVER inherit - see the
                # Parakeet subprocess plugin's docstring for exactly why
                # (a chatty worker's logging can corrupt LiveKit's own
                # internal IPC if it shares the parent's stdio directly).
                start_new_session=True,
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
                    f"Chatterbox worker did not report ready within "
                    f"{WORKER_INIT_TIMEOUT_SECS:.0f}s - run chatterbox_worker.py "
                    f"standalone in its venv to debug."
                )
            if not ready_line:
                self._process = None
                raise RuntimeError(
                    "Chatterbox worker exited before reporting ready - check "
                    "stderr above (likely a chatterbox/torch import error, or "
                    "the pkuseg install issue noted in the worker's docstring)."
                )
            payload = json.loads(ready_line.decode("utf-8"))
            if not payload.get("ready"):
                self._process = None
                raise RuntimeError(
                    f"Chatterbox worker init failed: {payload.get('error')}"
                )
            logger.info("ChatterboxSubprocessTTS: worker ready (warm).")

    async def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            async for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug(f"[chatterbox_worker stderr] {text}")
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
                logger.warning(
                    "ChatterboxSubprocessTTS: worker did not exit after kill()"
                )
        self._process = None

    def synthesize(self, text: str, *, conn_options=None) -> "ChatterboxChunkedStream":
        return ChatterboxChunkedStream(
            tts=self, input_text=text, conn_options=conn_options
        )


class ChatterboxChunkedStream(tts.ChunkedStream):
    """Non-streaming synthesis wrapper (see module docstring's honest
    limitation note) - one full utterance in, one full utterance's audio
    out, delivered as multiple transport chunks."""

    def __init__(
        self, *, tts: ChatterboxSubprocessTTS, input_text: str, conn_options=None
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter) -> None:
        await self._tts._ensure_started()
        proc = self._tts._process

        req_id = uuid.uuid4().hex
        self._tts._current_req_id = req_id

        output_emitter.initialize(
            request_id=req_id,
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        async with self._tts._request_lock:
            cmd = {"action": "tts", "id": req_id, "text": self.input_text}
            try:
                proc.stdin.write((json.dumps(cmd) + "\n").encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._tts._process = None
                raise RuntimeError(f"Chatterbox worker pipe broken: {exc}") from exc

            while True:
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=WORKER_READ_TIMEOUT_SECS
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"ChatterboxSubprocessTTS: no output for "
                        f"{WORKER_READ_TIMEOUT_SECS}s (id={req_id}), killing worker"
                    )
                    await self._tts._cleanup_process()
                    raise RuntimeError("Chatterbox worker timed out")

                if not line:
                    self._tts._process = None
                    raise RuntimeError("Chatterbox worker exited unexpectedly")

                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                if data.get("id") != req_id:
                    logger.debug(
                        f"ChatterboxSubprocessTTS: discarding stale line for "
                        f"id={data.get('id')} (current id={req_id})"
                    )
                    continue

                if data.get("done"):
                    break
                if "error" in data:
                    raise RuntimeError(f"Chatterbox worker error: {data['error']}")
                if "audio_b64" in data:
                    pcm_bytes = base64.b64decode(data["audio_b64"])
                    output_emitter.push(pcm_bytes)

        output_emitter.flush()
