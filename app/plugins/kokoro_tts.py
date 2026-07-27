"""
Kokoro TTS as a LiveKit TTS plugin - subprocess edition, real streaming.

Same architecture as plugins/chatterbox_tts.py (warmed via `await` inside
entrypoint(), never via prewarm(proc)'s throwaway-loop context - see that
file's docstring for exactly why that matters), and the same "never
inherit stderr" fix from plugins/parakeet_stt_subprocess.py. The one real
difference: kokoro_worker.py streams audio_b64 messages AS SEGMENTS ARE
SYNTHESIZED (a true generator underneath), not all-at-once after a single
monolithic call - so this plugin can start forwarding audio to the caller
meaningfully earlier for multi-sentence responses. Measure this via the
dev console's latency comparison rather than assuming the difference is
large in practice - same discipline as the whisper-vs-Parakeet finding.

ENV VARS:
  KOKORO_PYTHON - path to the dedicated venv's python
  KOKORO_WORKER - path to kokoro_worker.py

VOICE SELECTION: system_config.json's tts.model field IS the Kokoro voice
name (e.g. "af_heart") - passed straight through to the worker's "voice"
field per request.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid

from loguru import logger

from livekit.agents import tts

KOKORO_SAMPLE_RATE = 24000
WORKER_READ_TIMEOUT_SECS = 30.0
WORKER_INIT_TIMEOUT_SECS = 180.0


class KokoroSubprocessTTS(tts.TTS):
    def __init__(
        self,
        *,
        python_exec: str | None = None,
        worker_script: str | None = None,
        voice: str = "af_heart",
        sample_rate: int = KOKORO_SAMPLE_RATE,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._python_exec = python_exec or os.environ["KOKORO_PYTHON"]
        self._worker_script = worker_script or os.environ["KOKORO_WORKER"]
        self._voice = voice
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
                f"KokoroSubprocessTTS: spawning worker via {self._python_exec} ..."
            )
            self._process = await asyncio.create_subprocess_exec(
                self._python_exec,
                self._worker_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                limit=10 * 1024 * 1024,
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
                    f"Kokoro worker did not report ready within "
                    f"{WORKER_INIT_TIMEOUT_SECS:.0f}s - run kokoro_worker.py "
                    f"standalone in its venv to debug."
                )
            if not ready_line:
                self._process = None
                raise RuntimeError(
                    "Kokoro worker exited before reporting ready - check stderr above."
                )
            payload = json.loads(ready_line.decode("utf-8"))
            if not payload.get("ready"):
                self._process = None
                raise RuntimeError(f"Kokoro worker init failed: {payload.get('error')}")
            logger.info("KokoroSubprocessTTS: worker ready (warm).")

    async def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            async for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug(f"[kokoro_worker stderr] {text}")
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
                logger.warning("KokoroSubprocessTTS: worker did not exit after kill()")
        self._process = None

    def synthesize(self, text: str, *, conn_options=None) -> "KokoroChunkedStream":
        return KokoroChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class KokoroChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: KokoroSubprocessTTS, input_text: str, conn_options=None):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter) -> None:
        await self._tts._ensure_started()
        proc = self._tts._process

        req_id = uuid.uuid4().hex
        output_emitter.initialize(
            request_id=req_id,
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        async with self._tts._request_lock:
            cmd = {
                "action": "tts",
                "id": req_id,
                "text": self.input_text,
                "voice": self._tts._voice,
            }
            try:
                proc.stdin.write((json.dumps(cmd) + "\n").encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._tts._process = None
                raise RuntimeError(f"Kokoro worker pipe broken: {exc}") from exc

            while True:
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=WORKER_READ_TIMEOUT_SECS
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"KokoroSubprocessTTS: no output for "
                        f"{WORKER_READ_TIMEOUT_SECS}s (id={req_id}), killing worker"
                    )
                    await self._tts._cleanup_process()
                    raise RuntimeError("Kokoro worker timed out")

                if not line:
                    self._tts._process = None
                    raise RuntimeError("Kokoro worker exited unexpectedly")

                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                if data.get("id") != req_id:
                    logger.debug(
                        f"KokoroSubprocessTTS: discarding stale line for "
                        f"id={data.get('id')} (current id={req_id})"
                    )
                    continue

                if data.get("done"):
                    break
                if "error" in data:
                    raise RuntimeError(f"Kokoro worker error: {data['error']}")
                if "audio_b64" in data:
                    output_emitter.push(base64.b64decode(data["audio_b64"]))

        output_emitter.flush()
