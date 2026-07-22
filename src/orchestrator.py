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
import signal
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


def get_max_concurrent_calls() -> int:
    """Concurrent-call admission ceiling, read from the confirmed STT entry's max_concurrent_calls
    (see config/models_config.json). Since STT is now a shared microservice (not a per-call model),
    this is the shared services' throughput ceiling rather than "how many Whisper copies fit in
    VRAM" — set it from real load-testing. Defaults to 1 if nothing's confirmed yet or the entry
    omits the field, so an unconfigured/misconfigured cap fails safe (rejects) rather than
    over-admits."""
    if STATE.selection is None:
        return 1
    cfg = load_models_config()
    entry = find_entry(cfg, "stt", STATE.selection.get("stt", ""))
    if entry is None:
        return 1
    return int(entry.get("max_concurrent_calls", 1))


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
        # Sequential, not parallel: concurrent CUDA init on this box (vLLM loading a 14B model at
        # the same time as the whisper service's own CUDA context/model load) contends badly on
        # the GPU/PCIe — proven in practice to blow past startup_timeout_s. Load one at a time;
        # total wall-clock is the sum instead of the max, but it's the safer trade on this
        # hardware. Order: the three UI-backed services (llm/stt/tts), each flipping its own
        # backend row to "ready", then the worker last (no UI row — its "Prewarm complete." gates
        # overall phase="ready"). The worker's prewarm (Silero VAD + RAG index, no GPU) has no
        # dependency on the services being up, and the frontend never reveals "Start Call" until
        # phase == "ready", so no real call reaches entrypoint() before every service is reachable.
        await _launch_vllm(llm_entry)
        await _launch_whisper(stt_entry)
        await _launch_chatterbox(tts_entry)
        await _launch_worker(llm_entry, stt_entry, tts_entry)
        STATE.phase = "ready"
    except Exception as e:
        STATE.phase = "error"
        STATE.error = str(e)
        await _kill_remaining()


async def _spawn(cmd: list[str], env: dict, log_path: Path) -> asyncio.subprocess.Process:
    logf = open(log_path, "wb")
    # start_new_session=True puts the child in its own process group (pgid == child pid). vLLM
    # in particular forks a separate "VLLM::EngineCore" child that holds ALL the GPU memory and
    # does NOT die when only its parent is signalled — a plain terminate() on the parent leaves
    # that EngineCore orphaned (reparented to init), still holding ~13GB, which then makes the
    # next vLLM launch fail with "Engine core initialization failed". Killing the whole process
    # group (see _terminate_group) takes the EngineCore down with the parent. Same protection
    # for the agent worker, which also spawns its own job-executor subprocesses.
    return await asyncio.create_subprocess_exec(
        *cmd, cwd=str(PROJECT_ROOT), env=env, stdout=logf, stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )


def _terminate_group(proc: asyncio.subprocess.Process, sig: int = signal.SIGTERM) -> None:
    """Signal the child's entire process group, so children it spawned (vLLM's EngineCore, the
    worker's job executors) go down with it rather than orphaning and holding the GPU."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except ProcessLookupError:
        pass  # already gone


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
            _terminate_group(proc, signal.SIGKILL)  # group, so a stuck vLLM's EngineCore dies too
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


async def _launch_whisper(entry: dict) -> None:
    """Launch the shared faster-whisper STT microservice (src/whisper_server.py) and wait for its
    HTTP /health. Owns the "stt" backend row (previously stood in for by the worker's prewarm).
    The model settings go in as env vars — the service reads them at module scope."""
    env = os.environ.copy()
    env["WHISPER_MODEL_SIZE"] = entry["model_size"]
    env["WHISPER_COMPUTE_TYPE"] = entry["compute_type"]
    env["WHISPER_DEVICE"] = entry["device"]
    env["WHISPER_DEVICE_INDEX"] = str(entry["device_index"])
    env["WHISPER_LANGUAGE"] = entry["language"]
    env["WHISPER_NUM_WORKERS"] = str(entry.get("num_workers", 2))
    env["WHISPER_PORT"] = str(entry["port"])
    # faster-whisper/CTranslate2 borrows a CUDA lib dir on this box (same as the old worker did).
    if entry.get("cuda_lib_dir"):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [entry["cuda_lib_dir"], env.get("LD_LIBRARY_PATH", "")]
        )

    cmd = [str(VENV_PY), "-u", "src/whisper_server.py"]
    log_path = PROJECT_ROOT / entry["log_file"]
    proc = await _spawn(cmd, env, log_path)
    STATE.procs["stt"] = proc
    await _poll_until(
        lambda: _http_ready(entry["port"], entry["readiness"]["path"]),
        proc, log_path, entry.get("startup_timeout_s", 90), "Whisper service",
    )
    STATE.backends["stt"] = BackendState("ready")


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
    """Launch the LiveKit agent worker. It reaches the three services over HTTP (URLs below), so
    its own readiness is just prewarm (Silero VAD + RAG index, no GPU model) — it has no UI
    backend row of its own; "Prewarm complete." simply gates overall phase="ready" (set in _run)."""
    env = os.environ.copy()
    env["VLLM_URL"] = f"http://localhost:{llm_entry['port']}/v1"
    env["VLLM_MODEL"] = llm_entry["served_model_name"]
    env["CHATTERBOX_URL"] = f"http://localhost:{tts_entry['port']}{tts_entry['url_path']}"
    env["WHISPER_URL"] = f"http://localhost:{stt_entry['port']}{stt_entry['url_path']}"
    env["WHISPER_LANGUAGE"] = stt_entry["language"]

    cmd = [str(VENV_PY), "-u", "src/worker.py", "start"]
    log_path = LOGS_DIR / "worker.log"
    proc = await _spawn(cmd, env, log_path)
    STATE.procs["worker"] = proc
    await _poll_until(
        lambda: _log_grep_ready(log_path, "Prewarm complete."),
        proc, log_path, stt_entry.get("startup_timeout_s", 90), "Agent worker",
    )


async def _kill_remaining() -> None:
    for proc in STATE.procs.values():
        _terminate_group(proc)


async def shutdown_all() -> None:
    """Called from token_server.py's FastAPI shutdown hook — the Python analog of run.sh's
    cleanup() trap. Terminates worker first so it deregisters from LiveKit cleanly before its
    backends (vLLM/Chatterbox) disappear out from under it. Signals whole process groups, not
    just the direct children, so vLLM's EngineCore (which holds the GPU) can't be orphaned."""
    for name in ("worker", "tts", "stt", "llm"):
        proc = STATE.procs.get(name)
        if proc:
            _terminate_group(proc)
    for proc in STATE.procs.values():
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            _terminate_group(proc, signal.SIGKILL)


def get_status() -> dict:
    return {
        "phase": STATE.phase,
        "backends": {k: {"status": v.status, "detail": v.detail} for k, v in STATE.backends.items()},
        "selection": STATE.selection,
        "error": STATE.error,
    }
