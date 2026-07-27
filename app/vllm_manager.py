"""
vllm_manager.py - controls the vLLM subprocess so a developer can switch
LLM models from the dev UI, with an HONEST accounting of what that costs.

WHAT THIS DOES NOT PRETEND: switching the LLM model is NOT instant like
switching TTS/STT backends. vLLM serves ONE model per running process.
Switching means: stop the current vLLM process (frees its ~8-12GB+ of
GPU memory), start a new one with the new model (which must load weights,
compile, and build its KV cache - the exact sequence that took 15-20+
seconds even in the BEST case we saw this week, and can take much longer
for a bigger model or a first-time Hugging Face download). During that
window, NO CALLS CAN USE THE LLM. This module makes that window as short
and as safe as possible, and reports real status throughout - it does not
hide the outage.

STATE MACHINE:
  IDLE -> STARTING -> HEALTHY  (normal boot)
  HEALTHY -> STOPPING -> IDLE -> STARTING -> HEALTHY  (a switch)
  any state -> FAILED  (health check timed out, or the process died)

HOW HEALTH IS CHECKED: poll GET {base_url}/v1/models until it returns 200
with the expected served_model_name in its data, or until a timeout. This
is the same endpoint you used by hand all week to check "what is vLLM
actually serving" - reusing it here means the health check asks the exact
question that mattered every time we debugged this manually.

SAFETY: only ONE switch can run at a time (an asyncio.Lock) - a second
"switch" request while one is in flight is rejected with a clear error
rather than spawning two overlapping vLLM processes and doubling GPU
usage, which would risk an OOM that takes down the one that WAS working.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from typing import Optional

import aiohttp

import system_config

# Matches the working values discovered this week (run_vllm.sh) - used as
# defaults for flags NOT specific to which model is loaded.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_MEM_UTIL = 0.40
DEFAULT_MAX_NUM_SEQS = 4

HEALTH_POLL_INTERVAL_SECS = 2.0
HEALTH_TIMEOUT_SECS = 180.0  # generous: covers cold model load + a possible
# first-time Hugging Face download for a "custom" model source


class VLLMManager:
    def __init__(
        self, *, vllm_python: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
    ) -> None:
        self._vllm_python = vllm_python  # path to the venv's python/vllm entrypoint
        self._host = host
        self._port = port
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self.state = "IDLE"  # IDLE | STARTING | HEALTHY | STOPPING | FAILED
        self.status_detail = "not started"
        self.current_served_name: Optional[str] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/v1"

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def _stop_current(self) -> None:
        if not self._process_alive():
            self.state = "IDLE"
            return
        self.state = "STOPPING"
        self.status_detail = "stopping current vLLM process..."
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=20)
        except asyncio.TimeoutError:
            self.status_detail = "vLLM did not exit gracefully, killing it"
            self._process.kill()
            await self._process.wait()
        self._process = None
        self.state = "IDLE"
        self.current_served_name = None

    async def _wait_healthy(self, served_model_name: str) -> bool:
        deadline = time.monotonic() + HEALTH_TIMEOUT_SECS
        async with aiohttp.ClientSession() as http:
            while time.monotonic() < deadline:
                if self._process is not None and self._process.returncode is not None:
                    self.status_detail = (
                        f"vLLM process exited early (code {self._process.returncode}) "
                        f"before becoming healthy - check its logs"
                    )
                    return False
                try:
                    async with http.get(
                        f"{self.base_url}/models",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            body = await resp.json()
                            names = [m.get("id") for m in body.get("data", [])]
                            if served_model_name in names:
                                return True
                            self.status_detail = (
                                f"vLLM is up but serving {names}, waiting for "
                                f"'{served_model_name}'..."
                            )
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    self.status_detail = "waiting for vLLM to accept connections..."
                await asyncio.sleep(HEALTH_POLL_INTERVAL_SECS)
        return False

    async def switch_model(
        self,
        *,
        source: str,
        served_model_name: str,
        extra_args: Optional[list[str]] = None,
    ) -> dict:
        """Stop whatever's running, start the requested model, wait for it
        to report healthy, and ONLY THEN update system_config so agent.py's
        next call picks it up. Returns a result dict; never raises for
        expected failure modes (timeout, bad model) - callers check
        result['ok']."""
        if self._lock.locked():
            return {"ok": False, "error": "A model switch is already in progress."}

        async with self._lock:
            t0 = time.monotonic()
            await self._stop_current()

            self.state = "STARTING"
            self.status_detail = f"starting vLLM with '{source}'..."
            # Real invocation mirrors run_vllm.sh's proven flags exactly,
            # substituting only the model source/served name.
            cmd = [
                self._vllm_python,
                "-m",
                "vllm",
                "serve",
                source,
                "--served-model-name",
                served_model_name,
                "--host",
                self._host,
                "--port",
                str(self._port),
                "--max-model-len",
                str(DEFAULT_MAX_MODEL_LEN),
                "--gpu-memory-utilization",
                str(DEFAULT_GPU_MEM_UTIL),
                "--max-num-seqs",
                str(DEFAULT_MAX_NUM_SEQS),
                "--kv-cache-dtype",
                "bfloat16",  # NOT fp8 - see run_vllm.sh:
                # this GPU class (compute capability 8.6) lacks native FP8
                # hardware support; fp8 here would repeat a real failure
                # from earlier this week.
            ]
            if extra_args:
                cmd.extend(extra_args)

            try:
                self._process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                )
            except (OSError, FileNotFoundError) as exc:
                self.state = "FAILED"
                self.status_detail = f"failed to launch vLLM: {exc}"
                return {"ok": False, "error": self.status_detail}

            healthy = await self._wait_healthy(served_model_name)
            elapsed = time.monotonic() - t0

            if not healthy:
                self.state = "FAILED"
                self.status_detail = (
                    f"{self.status_detail} (timed out after {elapsed:.0f}s)"
                )
                return {
                    "ok": False,
                    "error": self.status_detail,
                    "elapsed_secs": elapsed,
                }

            self.state = "HEALTHY"
            self.current_served_name = served_model_name
            self.status_detail = f"healthy, serving '{served_model_name}'"

            # Only update the shared config AFTER confirming vLLM is
            # actually healthy - agent.py must never be told to use a
            # model that isn't really running.
            cfg = system_config.get_config()
            cfg["llm"] = {
                "served_model_name": served_model_name,
                "source": source,
                "display_name": cfg.get("llm", {}).get(
                    "display_name", served_model_name
                ),
            }
            system_config.save_config(cfg)

            return {
                "ok": True,
                "elapsed_secs": elapsed,
                "served_model_name": served_model_name,
            }

    def status(self) -> dict:
        return {
            "state": self.state,
            "detail": self.status_detail,
            "current_served_name": self.current_served_name,
            "switching": self._lock.locked(),
        }
