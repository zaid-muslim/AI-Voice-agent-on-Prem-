"""
Qwen3-TTS subprocess worker as a LiveKit TTS plugin.

REUSES YOUR EXISTING qwen_worker.py UNCHANGED - same stdin/stdout JSON
protocol, same isolated .venv-voice interpreter. Only the Pipecat-side
bridge (QwenTTSService) is replaced by this LiveKit equivalent.

Every hard-won fix from the Pipecat bridge is preserved, mapped 1:1:

  Pipecat bridge fix                      -> here
  --------------------------------------------------------------------------
  start_new_session=True on spawn         -> identical (Ctrl+C in the agent
                                             terminal must never SIGINT the
                                             worker mid-CUDA-call; that was
                                             the source of next-run OOMs)
  worker keeps SIG_IGN for SIGINT         -> unchanged, it's in YOUR worker
  30s read timeout (wedged-CUDA guard)    -> WORKER_READ_TIMEOUT_SECS
  per-request id tagging + stale-line     -> identical protocol; stale lines
    discard (interruption bleed-through)     logged at DEBUG, not swallowed
  cancel command on interruption          -> sent from _run()'s CancelledError
                                             path (LiveKit cancels the stream
                                             task on interruption)
  respawn-on-next-turn after crash        -> _ensure_worker() at stream start
  init warm-up (3 full runs, ~245ms       -> unchanged, it's in YOUR worker;
    first-chunk floor from turn one)         first agent job pays it once

New here (LiveKit-specific):
  - One request at a time via _request_lock: the worker's stdout is a single
    shared pipe, so exactly one ChunkedStream may own send+read at any
    moment. The Pipecat bridge had the same single-reader property
    implicitly (sequential run_tts calls); here it's explicit.
  - AgentSession wraps this non-streaming-input TTS with its sentence
    tokenizer StreamAdapter automatically, so LLM tokens still become
    speech sentence-by-sentence (same effective behavior as before).

VERIFY AGAINST YOUR INSTALLED livekit-agents (see README): ChunkedStream's
_run(output_emitter) signature and AudioEmitter method names are the most
drift-prone API surface in this project.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid

from loguru import logger

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    tts,
    utils,
)

WORKER_READ_TIMEOUT_SECS = 30.0
WORKER_INIT_TIMEOUT_SECS = 120.0


class QwenSubprocessTTS(tts.TTS):
    def __init__(
        self,
        *,
        python_exec: str | None = None,
        worker_script: str | None = None,
        model_id: str | None = None,
        speaker: str = "aiden",
        language: str = "English",
        chunk_size: int = 8,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        # Paths come from env so nothing user-specific is hardcoded here.
        self._python_exec = python_exec or os.environ.get(
            "QWEN_PYTHON", "/home/nauyan/voice-agent-pipeline/.venv-voice/bin/python"
        )
        self._worker_script = worker_script or os.environ.get(
            "QWEN_WORKER",
            "/home/nauyan/voice-agent-pipeline/src_2/services/qwen_worker.py",
        )
        self._model_id = model_id or os.environ.get(
            "QWEN_MODEL_ID", "/home/nauyan/voice-agent-pipeline/models/Qwentts"
        )
        self._speaker = os.environ.get("QWEN_SPEAKER", speaker)
        self._language = os.environ.get("QWEN_LANGUAGE", language)
        self._chunk_size = chunk_size

        self._process: asyncio.subprocess.Process | None = None
        self._spawn_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()  # one stream owns the pipe at a time
        self._current_req_id: str | None = None

    # ----------------------------------------------------------- lifecycle
    def _process_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def _spawn_worker(self) -> None:
        logger.info(f"QwenSubprocessTTS: spawning worker via {self._python_exec} ...")
        self._process = await asyncio.create_subprocess_exec(
            self._python_exec,
            self._worker_script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # inherit: worker tracebacks + timing lines stay visible
            # CRITICAL (ported): detach from the terminal's process group so
            # Ctrl+C never reaches the worker directly. The worker only dies
            # via our aclose() path, which tears CUDA down deterministically.
            start_new_session=True,
        )
        init_cmd = {
            "action": "init",
            "model_id": self._model_id,
            "speaker": self._speaker,
            "language": self._language,
            "chunk_size": self._chunk_size,
        }
        self._process.stdin.write((json.dumps(init_cmd) + "\n").encode())
        await self._process.stdin.drain()
        try:
            ready = await asyncio.wait_for(
                self._process.stdout.readline(), timeout=WORKER_INIT_TIMEOUT_SECS
            )
        except asyncio.TimeoutError:
            self._process.kill()
            self._process = None
            raise APIConnectionError(
                "Qwen TTS worker init timed out - run qwen_worker.py standalone to debug"
            )
        if not ready:
            self._process = None
            raise APIConnectionError(
                "Qwen TTS worker exited before reporting ready - check stderr above"
            )
        logger.info("QwenSubprocessTTS: worker ready (warm).")

    async def _ensure_worker(self) -> None:
        if not self._process_alive():
            async with self._spawn_lock:
                if not self._process_alive():
                    await self._spawn_worker()

    async def prewarm(self) -> None:
        """Spawn + warm the worker before the first call (used by agent.py)."""
        await self._ensure_worker()

    async def _send_cancel(self, req_id: str) -> None:
        if self._process_alive():
            try:
                self._process.stdin.write(
                    (json.dumps({"action": "cancel", "id": req_id}) + "\n").encode()
                )
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass  # worker's gone; next stream respawns it

    async def _kill_worker(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.kill()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning("QwenSubprocessTTS: worker did not exit after kill()")
        self._process = None

    async def aclose(self) -> None:
        await self._kill_worker()
        await super().aclose()

    # ------------------------------------------------------------ synthesize
    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "_QwenChunkedStream":
        return _QwenChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _QwenChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: QwenSubprocessTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._qtts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        q = self._qtts
        req_id = uuid.uuid4().hex
        output_emitter.initialize(
            request_id=req_id,
            sample_rate=q.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        async with q._request_lock:  # exactly one owner of the pipe
            await q._ensure_worker()
            q._current_req_id = req_id
            t_sent = time.perf_counter()
            try:
                q._process.stdin.write(
                    (
                        json.dumps(
                            {"action": "tts", "id": req_id, "text": self.input_text}
                        )
                        + "\n"
                    ).encode()
                )
                await q._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                q._process = None
                raise APIConnectionError(f"Qwen TTS worker pipe broken: {exc}")

            first_chunk = True
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(
                            q._process.stdout.readline(),
                            timeout=WORKER_READ_TIMEOUT_SECS,
                        )
                    except asyncio.TimeoutError:
                        # Wedged CUDA guard (ported): kill it; the next
                        # stream respawns a fresh, warm worker.
                        logger.error(
                            f"QwenSubprocessTTS: no output for "
                            f"{WORKER_READ_TIMEOUT_SECS}s (id={req_id}), killing worker"
                        )
                        await q._kill_worker()
                        raise APIConnectionError("Qwen TTS worker timed out")

                    if not line:
                        q._process = None
                        raise APIConnectionError("Qwen TTS worker exited unexpectedly")

                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    if data.get("id") != req_id:
                        # Tail of an interrupted request - visible, not silent.
                        logger.debug(
                            f"QwenSubprocessTTS: discarding stale line for "
                            f"id={data.get('id')} (current id={req_id})"
                        )
                        continue

                    if data.get("done"):
                        break
                    if "error" in data:
                        raise APIConnectionError(f"Qwen TTS error: {data['error']}")
                    if "audio_b64" in data:
                        if first_chunk:
                            first_chunk = False
                            logger.info(
                                f"QwenSubprocessTTS timing: id={req_id} round trip "
                                f"to first audio chunk = "
                                f"{(time.perf_counter() - t_sent) * 1000:.1f}ms "
                                f"(compare against the worker's stderr timing line)"
                            )
                        output_emitter.push(base64.b64decode(data["audio_b64"]))

                output_emitter.flush()

            except asyncio.CancelledError:
                # LiveKit cancels this task on interruption. Tell the worker
                # to stop burning GPU on audio nobody will hear (ported from
                # _handle_interruption), then let cancellation propagate.
                # The worker replies with leftover lines tagged with this
                # req_id; the NEXT stream discards them as stale - by design.
                await q._send_cancel(req_id)
                raise
