#!/usr/bin/env python3
"""Post-confirmation backend orchestration for the model-selection feature.

Launches vLLM, the shared Whisper STT service, Chatterbox, and the agent worker as Docker
containers (via `docker compose`) per the user's confirmed config/models_config.json selection,
and tracks readiness for the frontend to poll. Each of the four lives in docker-compose.yml under
the "on-demand" profile — nothing here starts at pipeline boot, only after a browser confirms a
selection, matching the pre-Docker subprocess-based design this replaces.

Every `docker compose` invocation touching these four services passes `--profile on-demand`
explicitly, on every subcommand (up/stop/logs/ps) — Compose has a known inconsistency where naming
a profiled service alone doesn't reliably activate its profile for every subcommand, so don't rely
on that; always pass the flag.
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

CATEGORIES = ("llm", "stt", "tts")
# UI category -> docker-compose.yml service name. The worker has no UI category/backend row of
# its own (see BackendState below) but is launched the same way as these three.
CATEGORY_TO_SERVICE = {"llm": "vllm", "stt": "whisper", "tts": "chatterbox"}
COMPOSE_BASE = ["docker", "compose", "--profile", "on-demand"]


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
    # Background `docker compose logs -f` tailers (logs/<service>.log), for debugging convenience
    # only — not the container's actual lifecycle, which docker compose itself owns. Best-effort:
    # losing one doesn't affect correctness, just makes `tail logs/whisper.log`-style debugging
    # unavailable for that service.
    log_tailers: dict[str, asyncio.subprocess.Process] = field(default_factory=dict)


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
        STATE.log_tailers = {}

    asyncio.create_task(_run(entries["llm"], entries["stt"], entries["tts"]))


async def _run(llm_entry: dict, stt_entry: dict, tts_entry: dict) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    try:
        # Sequential, not parallel: concurrent CUDA init on this box (vLLM loading a 14B model at
        # the same time as the whisper service's own CUDA context/model load) contends badly on
        # the GPU/PCIe — proven in practice to blow past startup_timeout_s. Load one at a time;
        # total wall-clock is the sum instead of the max, but it's the safer trade on this
        # hardware. Order: the three UI-backed services (llm/stt/tts), each flipping its own
        # backend row to "ready", then the worker last (no UI row, no GPU of its own — its
        # readiness sentinel gates overall phase="ready"). The frontend never reveals "Start Call"
        # until phase == "ready", so no real call reaches entrypoint() before every service is up,
        # regardless of launch order.
        await _launch_vllm(llm_entry)
        await _launch_whisper(stt_entry)
        await _launch_chatterbox(tts_entry)
        await _launch_worker(llm_entry, stt_entry, tts_entry)
        STATE.phase = "ready"
    except Exception as e:
        STATE.phase = "error"
        STATE.error = str(e)
        await _kill_remaining()


async def _compose_up(service: str, log_path: Path, env_overrides: dict[str, str] | None = None) -> None:
    """`docker compose up -d --build <service>` — blocks until the image is built (if needed) and
    the container is created+started, but not until the app inside is actually ready (that's
    _poll_until's job, via HTTP or Docker health status). Raises RuntimeError with the command's
    own output on a hard failure (bad Dockerfile, image pull failure, Docker daemon down, etc.) —
    distinct from the app-level failures _poll_until watches for once the container is running."""
    env = os.environ.copy()
    env.update(env_overrides or {})
    proc = await asyncio.create_subprocess_exec(
        *COMPOSE_BASE, "up", "-d", "--build", service,
        cwd=str(PROJECT_ROOT), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker compose up failed for {service} (code {proc.returncode}): "
            f"{out.decode(errors='replace')[-2000:]}"
        )
    await _start_log_tailer(service, log_path)


async def _start_log_tailer(service: str, log_path: Path) -> None:
    """Best-effort: mirror the container's logs into logs/<service>.log for the same
    tail-the-log-file debugging workflow every backend used pre-Docker. Losing this doesn't affect
    correctness — `docker compose logs <service>` always works as a fallback."""
    try:
        logf = open(log_path, "wb")
        tailer = await asyncio.create_subprocess_exec(
            *COMPOSE_BASE, "logs", "-f", "--no-color", "--since", "0s", service,
            cwd=str(PROJECT_ROOT), stdout=logf, stderr=asyncio.subprocess.STDOUT,
        )
        STATE.log_tailers[service] = tailer
    except OSError as e:
        print(f"  Warning: couldn't start log tailer for {service}: {e}")


async def _compose_container_id(service: str) -> str | None:
    # -a is required: `docker compose ps -q` without it silently omits exited containers, which
    # would make _compose_service_exited() below never actually detect a crash (it'd see "no
    # container id" and treat that as "not exited yet" instead of failing fast) — confirmed via a
    # real crashed container during vLLM testing, not a hypothetical.
    proc = await asyncio.create_subprocess_exec(
        *COMPOSE_BASE, "ps", "-a", "-q", service,
        cwd=str(PROJECT_ROOT), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    cid = out.decode().strip().splitlines()
    return cid[0] if cid else None


async def _docker_inspect_field(container_id: str, go_format: str) -> str | None:
    proc = await asyncio.create_subprocess_exec(
        "docker", "inspect", "--format", go_format, container_id,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    return out.decode().strip() or None


async def _http_ready(port: int, path: str) -> bool:
    # Matches run.sh's old `curl -s -o /dev/null URL` semantics: any successful HTTP response
    # counts as ready, no status-code check.
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://localhost:{port}{path}", timeout=aiohttp.ClientTimeout(total=2)
            ):
                return True
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False


async def _docker_health_ready(service: str) -> bool:
    """For the worker, which has no HTTP surface — readiness comes from Docker's own HEALTHCHECK
    (Dockerfile.worker: `test -f /tmp/prewarm-ready`, written by worker.py's prewarm())."""
    cid = await _compose_container_id(service)
    if cid is None:
        return False
    status = await _docker_inspect_field(cid, "{{.State.Health.Status}}")
    return status == "healthy"


async def _compose_service_exited(service: str) -> bool:
    """Dead-early detection — the Docker analog of the old `proc.returncode is not None` check.
    A container with no restart: policy that crashed sits in "exited" state; a container that
    hasn't been created yet (still building) reports no container id at all, which is NOT the
    same as exited — don't treat "not found yet" as a failure, only an actual exited state."""
    cid = await _compose_container_id(service)
    if cid is None:
        return False
    status = await _docker_inspect_field(cid, "{{.State.Status}}")
    return status == "exited"


async def _poll_until(ready_check, service: str, log_path: Path, timeout_s: float, what: str) -> None:
    """Docker analog of the old subprocess-based poll loop: watch for the container dying early,
    watch for readiness, time out and force-stop otherwise."""
    start = time.monotonic()
    while True:
        if await _compose_service_exited(service):
            raise RuntimeError(
                f"{what} exited early — see {log_path} (or `docker compose logs {service}`)"
            )
        if await ready_check():
            return
        if time.monotonic() - start > timeout_s:
            await _compose_stop(service, force=True)
            raise RuntimeError(f"{what} did not become ready within {timeout_s}s — see {log_path}")
        await asyncio.sleep(1)


async def _compose_stop(service: str, force: bool = False) -> None:
    cmd = [*COMPOSE_BASE, "kill" if force else "stop", service]
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


async def _launch_vllm(entry: dict) -> None:
    log_path = PROJECT_ROOT / entry["log_file"]
    await _compose_up("vllm", log_path)
    await _poll_until(
        lambda: _http_ready(entry["port"], entry["readiness"]["path"]),
        "vllm", log_path, entry.get("startup_timeout_s", 120), "vLLM",
    )
    STATE.backends["llm"] = BackendState("ready")


async def _launch_whisper(entry: dict) -> None:
    """The shared faster-whisper STT microservice — model settings flow from the confirmed config
    entry as env var overrides on the `docker compose up` call (see docker-compose.yml's whisper
    service, which reads them via ${VAR} interpolation)."""
    env_overrides = {
        "WHISPER_MODEL_SIZE": entry["model_size"],
        "WHISPER_COMPUTE_TYPE": entry["compute_type"],
        "WHISPER_DEVICE": entry["device"],
        "WHISPER_DEVICE_INDEX": str(entry["device_index"]),
        "WHISPER_LANGUAGE": entry["language"],
        "WHISPER_NUM_WORKERS": str(entry.get("num_workers", 2)),
    }
    log_path = PROJECT_ROOT / entry["log_file"]
    await _compose_up("whisper", log_path, env_overrides)
    await _poll_until(
        lambda: _http_ready(entry["port"], entry["readiness"]["path"]),
        "whisper", log_path, entry.get("startup_timeout_s", 90), "Whisper service",
    )
    STATE.backends["stt"] = BackendState("ready")


async def _launch_chatterbox(entry: dict) -> None:
    log_path = PROJECT_ROOT / entry["log_file"]
    await _compose_up("chatterbox", log_path)
    await _poll_until(
        lambda: _http_ready(entry["port"], entry["readiness"]["path"]),
        "chatterbox", log_path, entry.get("startup_timeout_s", 90), "Chatterbox",
    )
    STATE.backends["tts"] = BackendState("ready")


async def _launch_worker(llm_entry: dict, stt_entry: dict, tts_entry: dict) -> None:
    """Launch the LiveKit agent worker container. It reaches the three services over HTTP at
    127.0.0.1:<port> (host networking — see docker-compose.yml's worker service comment for why),
    so its own readiness is just prewarm (Silero VAD + RAG index, no GPU model) — it has no UI
    backend row of its own; reaching "healthy" simply gates overall phase="ready" (set in _run).

    127.0.0.1, not "localhost": confirmed live that this box's minimal container images can't
    resolve the literal string "localhost" (its /etc/hosts has no plain entry for it, and unlike
    the host itself they have no other NSS fallback) — 127.0.0.1 needs no name resolution at all.
    """
    env_overrides = {
        "VLLM_URL": f"http://127.0.0.1:{llm_entry['port']}/v1",
        "VLLM_MODEL": llm_entry["served_model_name"],
        "CHATTERBOX_URL": f"http://127.0.0.1:{tts_entry['port']}{tts_entry['url_path']}",
        "WHISPER_URL": f"http://127.0.0.1:{stt_entry['port']}{stt_entry['url_path']}",
        "WHISPER_LANGUAGE": stt_entry["language"],
    }
    log_path = LOGS_DIR / "worker.log"
    await _compose_up("worker", log_path, env_overrides)
    await _poll_until(
        lambda: _docker_health_ready("worker"), "worker", log_path,
        stt_entry.get("startup_timeout_s", 90), "Agent worker",
    )


async def _kill_remaining() -> None:
    for service in ("worker", "chatterbox", "whisper", "vllm"):
        await _compose_stop(service, force=True)


async def shutdown_all() -> None:
    """Called from token_server.py's FastAPI shutdown hook — the Python analog of run.sh's
    cleanup() trap. Stops worker first so it deregisters from LiveKit cleanly before its backends
    (vLLM/whisper/Chatterbox) disappear out from under it. Plain `docker compose stop` (SIGTERM,
    graceful) — no process-group-kill dance needed anymore: Docker's own cgroup-based container
    teardown can't leak children (e.g. vLLM's EngineCore) onto the host the way raw subprocess
    forking could, which is exactly the bug that workaround existed for pre-Docker."""
    for tailer in STATE.log_tailers.values():
        if tailer.returncode is None:
            tailer.terminate()
    for service in ("worker", "chatterbox", "whisper", "vllm"):
        await _compose_stop(service)


def get_status() -> dict:
    return {
        "phase": STATE.phase,
        "backends": {k: {"status": v.status, "detail": v.detail} for k, v in STATE.backends.items()},
        "selection": STATE.selection,
        "error": STATE.error,
    }
