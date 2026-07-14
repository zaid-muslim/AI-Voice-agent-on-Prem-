import asyncio
import json
import base64
import time
import uuid
from loguru import logger
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

# How long run_tts will wait for ANY line from the worker before declaring
# it wedged. Covers first-chunk synthesis + margin. A hung CUDA call in the
# worker previously stalled the whole pipeline silently and forever.
WORKER_READ_TIMEOUT_SECS = 30.0


class QwenTTSService(TTSService):
    """Bridge service that communicates with Qwen-TTS running as a subprocess
    inside the isolated .venv-voice.

    FIXES IN THIS REVISION:
    1. start_new_session=True on spawn. Ctrl+C in the terminal sends SIGINT
       to the ENTIRE foreground process group - the worker was receiving it
       directly and dying with a KeyboardInterrupt traceback (your
       qwen_worker.py:168 trace) before cancel()/_cleanup_process() ever
       ran. Uncontrolled CUDA teardown on that path is the most plausible
       remaining cause of OOM on the next run. With its own session, the
       worker's lifecycle is owned exclusively by this service.
    2. Read timeout in run_tts. readline() previously had no timeout: a
       wedged worker (CUDA hang) made the bot permanently, silently mute.
       Now it errors, kills the worker, and respawns on the next turn.
    3. Stale lines from interrupted requests are logged at DEBUG instead of
       silently dropped, so interruption bleed-through is visible if it
       ever recurs.

    (Previous revision's fixes retained: per-request id tagging, cancel
    forwarding on interruption, cancel() override for CancelFrame path,
    TTFB instrumentation paired with the worker's stderr timing lines.)
    """

    Settings = TTSSettings

    def __init__(
        self,
        model_id: str = "/home/nauyan/voice-agent-pipeline/models/Qwentts",
        speaker: str = "aiden",
        language: str = "English",
        chunk_size: int = 8,
        sample_rate: int = 24000,
        **kwargs,
    ):
        default_settings = self.Settings(
            model=model_id,
            voice=speaker,
            language=language,
        )
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )
        self._python_exec = "/home/nauyan/voice-agent-pipeline/.venv-voice/bin/python"
        self._worker_script = (
            "/home/nauyan/voice-agent-pipeline/src_2/services/qwen_worker.py"
        )
        self._model_id = model_id
        self._speaker = speaker
        self._language = language
        self._chunk_size = chunk_size
        self._process = None
        self._spawn_lock = asyncio.Lock()
        self._current_req_id = None  # id of the in-flight (or last) request

    async def _spawn_worker(self):
        """(Re)spawn the worker subprocess and wait for it to report ready."""
        logger.info(f"Spawning isolated TTS worker via {self._python_exec}...")
        self._process = await asyncio.create_subprocess_exec(
            self._python_exec,
            self._worker_script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # inherit parent's stderr so worker tracebacks and
            # timing lines aren't swallowed
            # CRITICAL: detach from the terminal's process group so Ctrl+C
            # (SIGINT to the foreground group) doesn't kill the worker
            # directly. The worker must only ever die via our
            # stop()/cancel() -> _cleanup_process() path, which tears CUDA
            # down deterministically.
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
            ready_line = await asyncio.wait_for(
                self._process.stdout.readline(), timeout=120
            )
        except asyncio.TimeoutError:
            logger.error("QwenTTSService: worker did not report ready within 120s")
            self._process.kill()
            self._process = None
            raise RuntimeError(
                "Qwen TTS worker init timed out - run qwen_worker.py standalone to debug"
            )

        if not ready_line:
            self._process = None
            raise RuntimeError(
                "Qwen TTS worker exited before reporting ready - check stderr above"
            )

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self, frame):
        await super().start(frame)
        if not self._process_alive():
            async with self._spawn_lock:
                if not self._process_alive():
                    await self._spawn_worker()

    async def _handle_interruption(self, frame, direction):
        # Tell the worker to stop generating the interrupted turn so it
        # stops burning GPU time on audio nobody will hear.
        if self._process_alive() and self._current_req_id is not None:
            try:
                cancel_cmd = {"action": "cancel", "id": self._current_req_id}
                self._process.stdin.write((json.dumps(cancel_cmd) + "\n").encode())
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass  # worker's already gone; run_tts will respawn it next turn
        await super()._handle_interruption(frame, direction)

    async def run_tts(self, text: str, context_id: str):
        # Respawn if the worker died since start() (e.g. it crashed on a
        # previous turn).
        if not self._process_alive():
            async with self._spawn_lock:
                if not self._process_alive():
                    try:
                        await self._spawn_worker()
                    except Exception as exc:
                        logger.error(f"QwenTTSService: failed to restart worker: {exc}")
                        yield ErrorFrame(f"Qwen TTS worker unavailable: {exc}")
                        return

        await self.start_ttfb_metrics()

        req_id = uuid.uuid4().hex
        self._current_req_id = req_id
        cmd = {"action": "tts", "id": req_id, "text": text}

        t_sent = time.perf_counter()
        try:
            self._process.stdin.write((json.dumps(cmd) + "\n").encode())
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            logger.error(f"QwenTTSService: worker pipe broken: {exc}")
            self._process = None  # force a respawn on the next turn
            yield ErrorFrame(f"Qwen TTS worker connection lost: {exc}")
            return

        first_chunk = True
        errored = False
        while True:
            # Timeout so a wedged worker can't mute the bot forever. On
            # timeout we kill it; the next turn's run_tts respawns fresh.
            try:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=WORKER_READ_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"QwenTTSService: no output from worker for "
                    f"{WORKER_READ_TIMEOUT_SECS}s (id={req_id}), killing it"
                )
                await self._cleanup_process()
                yield ErrorFrame("Qwen TTS worker timed out")
                break

            if not line:
                logger.error("QwenTTSService: worker stdout closed unexpectedly")
                self._process = None
                if not errored:
                    yield ErrorFrame("Qwen TTS worker exited unexpectedly")
                break
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            # Discard anything left over from a request we already moved on
            # from (e.g. the tail of an interrupted answer) - but visibly,
            # so bleed-through shows up in the logs if it recurs.
            if data.get("id") != req_id:
                logger.debug(
                    f"QwenTTSService: discarding stale line for "
                    f"id={data.get('id')} (current id={req_id})"
                )
                continue

            if data.get("done"):
                break
            if "error" in data:
                errored = True
                yield ErrorFrame(f"Qwen TTS Error: {data['error']}")
                continue
            if "audio_b64" in data:
                if first_chunk:
                    await self.stop_ttfb_metrics()
                    round_trip_ms = (time.perf_counter() - t_sent) * 1000
                    logger.info(
                        f"QwenTTSService timing: id={req_id} "
                        f"service-side round trip to first audio chunk = "
                        f"{round_trip_ms:.1f}ms (compare against the worker's "
                        f"'[worker timing] id={req_id}' stderr line; the gap "
                        f"is IPC overhead, the worker number is synthesis)"
                    )
                    first_chunk = False
                pcm_bytes = base64.b64decode(data["audio_b64"])
                yield TTSAudioRawFrame(
                    audio=pcm_bytes,
                    sample_rate=data.get("sample_rate", self.sample_rate),
                    num_channels=1,
                    context_id=context_id,
                )

    async def _cleanup_process(self):
        if self._process and self._process.returncode is None:
            self._process.kill()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning("QwenTTSService: worker did not exit after kill()")
        self._process = None

    async def stop(self, frame):
        # Normal graceful shutdown path (EndFrame).
        await self._cleanup_process()
        await super().stop(frame)

    async def cancel(self, frame):
        # Hard-cancellation path (CancelFrame). With start_new_session=True
        # this is now the ONLY way the worker dies on Ctrl+C - previously
        # SIGINT hit it directly and it crashed uncontrolled.
        await self._cleanup_process()
        await super().cancel(frame)
