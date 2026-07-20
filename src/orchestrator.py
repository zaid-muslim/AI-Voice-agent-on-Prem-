#!/usr/bin/env python3
"""Post-confirmation backend orchestration for the model-selection feature.

Launches vLLM, Chatterbox, and the agent worker as subprocesses per the user's confirmed
config/models_config.json selection, and tracks readiness for the frontend to poll. This is the
Python translation of run.sh's old backgrounding / `kill -0` liveness-check / readiness-polling
patterns, for the pieces that now start on demand (post-confirmation) instead of unconditionally
at pipeline boot — nothing here should silently diverge from that proven bash behavior (see the
`_http_ready`/`_log_grep_ready` comments below for the specific things that must match exactly).
"""
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_CONFIG_FILE = PROJECT_ROOT / "config" / "models_config.json"
LOGS_DIR = PROJECT_ROOT / "logs"
VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"

CATEGORIES = ("llm", "stt", "tts")


def load_models_config() -> dict:
    """Same soft-fail contract as worker.py's load_domain_config(): a missing or malformed
    file must never crash the token server, just leave the picker empty with a warning."""
    try:
        with open(MODELS_CONFIG_FILE, "r") as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"  Warning: couldn't load {MODELS_CONFIG_FILE}: {e}")
        return {c: [] for c in CATEGORIES}
    for c in CATEGORIES:
        cfg.setdefault(c, [])
    return cfg


def find_entry(cfg: dict, category: str, entry_id: str) -> dict | None:
    for entry in cfg.get(category, []):
        if entry["id"] == entry_id:
            return entry
    return None


@dataclass
class BackendState:
    status: str = "pending"  # pending | loading | ready | error
    detail: str | None = None


@dataclass
class OrchestratorState:
    phase: str = "idle"  # idle | loading | ready | error
    backends: dict[str, BackendState] = field(default_factory=dict)
    selection: dict | None = None
    error: str | None = None
    procs: dict[str, asyncio.subprocess.Process] = field(default_factory=dict)


STATE = OrchestratorState()
_start_lock = asyncio.Lock()


async def start_selection(selection: dict) -> None:
    """Validate the selection and kick off orchestration in the background. Raises ValueError
    for an unknown model id (caller maps to HTTP 400) or RuntimeError if a selection is already
    loading/ready (caller maps to HTTP 409) — re-selecting requires restarting the whole
    pipeline, per the per-run (not per-call) decision."""
    async with _start_lock:
        if STATE.phase in ("loading", "ready"):
            raise RuntimeError(
                f"pipeline already {STATE.phase} — restart the whole pipeline to choose differently"
            )
        cfg = load_models_config()
        entries = {c: find_entry(cfg, c, selection[c]) for c in CATEGORIES}
        missing = [c for c, e in entries.items() if e is None]
        if missing:
            raise ValueError(f"unknown model id for: {', '.join(missing)}")

        STATE.phase = "loading"
        STATE.selection = dict(selection)
        STATE.backends = {c: BackendState("loading") for c in CATEGORIES}
        STATE.error = None
        STATE.procs = {}

    asyncio.create_task(_run(entries["llm"], entries["stt"], entries["tts"]))


async def _run(llm_entry: dict, stt_entry: dict, tts_entry: dict) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    try:
        # Safe to launch concurrently: worker.py's prewarm() (Whisper/VAD) has no dependency on
        # vLLM/Chatterbox being up, and the frontend never reveals "Start Call" until phase ==
        # "ready" (all three green) — so no real call can reach entrypoint() before vLLM/
        # Chatterbox are actually ready, regardless of launch order. If this box's documented
        # CUDA finickiness (see run.sh's git history re: libnvrtc.so.13/libcublas.so.12) ever
        # makes concurrent CUDA init flaky in practice, switch this to sequential awaits.
        await asyncio.gather(
            _launch_vllm(llm_entry),
            _launch_chatterbox(tts_entry),
            _launch_worker(llm_entry, stt_entry, tts_entry),
        )
        STATE.phase = "ready"
    except Exception as e:
        STATE.phase = "error"
        STATE.error = str(e)
        await _kill_remaining()


async def _spawn(cmd: list[str], env: dict, log_path: Path) -> asyncio.subprocess.Process:
    logf = open(log_path, "wb")
    return await asyncio.create_subprocess_exec(
        *cmd, cwd=str(PROJECT_ROOT), env=env, stdout=logf, stderr=asyncio.subprocess.STDOUT,
    )


async def _http_ready(port: int, path: str) -> bool:
    # Matches run.sh's `curl -s -o /dev/null URL` semantics exactly: any successful HTTP
    # response counts as ready, no status-code check. Don't tighten this to `status == 200` —
    # that would diverge from the bash version's already-proven behavior.
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://localhost:{port}{path}", timeout=aiohttp.ClientTimeout(total=2)
            ):
                return True
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False


async def _log_grep_ready(log_path: Path, pattern: str) -> bool:
    try:
        return pattern in log_path.read_text()
    except FileNotFoundError:
        return False


async def _poll_until(ready_check, proc: asyncio.subprocess.Process, log_path: Path,
                       timeout_s: float, what: str) -> None:
    """Python analog of run.sh's `until <ready>; do if ! kill -0 $pid; then fail; fi; sleep; done`."""
    start = time.monotonic()
    while True:
        if proc.returncode is not None:
            raise RuntimeError(f"{what} exited early (code {proc.returncode}) — see {log_path}")
        if await ready_check():
            return
        if time.monotonic() - start > timeout_s:
            proc.kill()
            raise RuntimeError(f"{what} did not become ready within {timeout_s}s — see {log_path}")
        await asyncio.sleep(1)


async def _launch_vllm(entry: dict) -> None:
    env = os.environ.copy()
    env["CUDA_HOME"] = entry["cuda_home"]
    env["PATH"] = os.pathsep.join(
        [f"{entry['cuda_home']}/bin", *entry.get("extra_path_dirs", []), env.get("PATH", "")]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [f"{entry['cuda_home']}/lib", env.get("LD_LIBRARY_PATH", "")]
    )
    env.update(entry.get("extra_env", {}))

    cmd = [
        entry["vllm_bin"], "serve", entry["model"],
        "--served-model-name", entry["served_model_name"],
        *entry["launch_args"], "--port", str(entry["port"]),
    ]
    log_path = PROJECT_ROOT / entry["log_file"]
    proc = await _spawn(cmd, env, log_path)
    STATE.procs["llm"] = proc
    await _poll_until(
        lambda: _http_ready(entry["port"], entry["readiness"]["path"]),
        proc, log_path, entry.get("startup_timeout_s", 120), "vLLM",
    )
    STATE.backends["llm"] = BackendState("ready")


async def _launch_chatterbox(entry: dict) -> None:
    cmd = [entry["python_bin"], "-u", entry["script"]]
    log_path = PROJECT_ROOT / entry["log_file"]
    proc = await _spawn(cmd, os.environ.copy(), log_path)
    STATE.procs["tts"] = proc
    await _poll_until(
        lambda: _log_grep_ready(log_path, entry["readiness"]["pattern"]),
        proc, log_path, entry.get("startup_timeout_s", 90), "Chatterbox",
    )
    STATE.backends["tts"] = BackendState("ready")


async def _launch_worker(llm_entry: dict, stt_entry: dict, tts_entry: dict) -> None:
    env = os.environ.copy()
    env["VLLM_URL"] = f"http://localhost:{llm_entry['port']}/v1"
    env["VLLM_MODEL"] = llm_entry["served_model_name"]
    env["CHATTERBOX_URL"] = f"http://localhost:{tts_entry['port']}{tts_entry['url_path']}"
    env["WHISPER_MODEL_SIZE"] = stt_entry["model_size"]
    env["WHISPER_COMPUTE_TYPE"] = stt_entry["compute_type"]
    env["WHISPER_DEVICE"] = stt_entry["device"]
    env["WHISPER_DEVICE_INDEX"] = str(stt_entry["device_index"])
    env["WHISPER_LANGUAGE"] = stt_entry["language"]
    if stt_entry.get("cuda_lib_dir"):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [stt_entry["cuda_lib_dir"], env.get("LD_LIBRARY_PATH", "")]
        )

    cmd = [str(VENV_PY), "-u", "src/worker.py", "start"]
    log_path = LOGS_DIR / "worker.log"
    proc = await _spawn(cmd, env, log_path)
    STATE.procs["worker"] = proc
    # This single check stands in for the whole worker subprocess's readiness (VAD + Whisper +
    # RAG index all load in prewarm() before this string prints) — matches the "stt" category in
    # the UI since that's the only category the worker process itself gates on loading.
    await _poll_until(
        lambda: _log_grep_ready(log_path, "Prewarm complete."),
        proc, log_path, stt_entry.get("startup_timeout_s", 90), "Agent worker",
    )
    STATE.backends["stt"] = BackendState("ready")


async def _kill_remaining() -> None:
    for proc in STATE.procs.values():
        if proc.returncode is None:
            proc.terminate()


async def shutdown_all() -> None:
    """Called from token_server.py's FastAPI shutdown hook — the Python analog of run.sh's
    cleanup() trap. Terminates worker first so it deregisters from LiveKit cleanly before its
    backends (vLLM/Chatterbox) disappear out from under it."""
    for name in ("worker", "tts", "llm"):
        proc = STATE.procs.get(name)
        if proc and proc.returncode is None:
            proc.terminate()
    for proc in STATE.procs.values():
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()


def get_status() -> dict:
    return {
        "phase": STATE.phase,
        "backends": {k: {"status": v.status, "detail": v.detail} for k, v in STATE.backends.items()},
        "selection": STATE.selection,
        "error": STATE.error,
    }
