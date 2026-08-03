"""STT/TTS backend construction, ported unchanged (already fully
domain-agnostic) from ``app/main.py:130-381,1115-1183``
(``_make_stt``/``_make_tts``/reachability checks/warm-ups). Reused via a
fresh, generalized copy rather than importing ``app/main.py`` directly -
see ``domain_agent_core``'s architecture plan for why (independent
pipelines carrying their own copies of generic mechanisms is this
codebase's own established convention, e.g. ``app/prompts.py``'s
``HOSPITAL_TZ`` duplication comment).

Every STT/TTS plugin class imported here comes from ``app/plugins/*``
(read-only import - those plugin modules have no hospital-specific
content, confirmed by direct reading) so this file does not duplicate any
actual plugin implementation, only the selection/fallback logic around it.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from loguru import logger

_APP_DIR = Path(__file__).resolve().parents[2] / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "distil-large-v3")
SHARED_STT_BASE_URL = os.environ.get("SHARED_STT_BASE_URL", "http://localhost:8020")
SHARED_TTS_BASE_URL = os.environ.get("SHARED_TTS_BASE_URL", "http://localhost:8021")
QWEN_OMNI_MODEL = os.environ.get(
    "QWEN_OMNI_MODEL",
    str(_APP_DIR.parent / "models" / "qwen3-tts" / "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
)
QWEN_OMNI_BASE_URL = os.environ.get("QWEN_OMNI_BASE_URL", "http://192.168.18.56:8091/v1")
QWEN_OMNI_VOICE = os.environ.get("QWEN_OMNI_VOICE", "Aiden")


def _tcp_reachable(base_url: str, label: str, timeout: float = 2.0) -> bool:
    """Fast TCP-level reachability check, run once per worker process at
    prewarm time - not per call. A real HTTP health check still happens
    afterward via the plugin's own ``.load()``/warm-up call.

    Args:
        base_url: The service's base URL.
        label: Short description used in the log line on failure.
        timeout: Socket connect timeout in seconds.

    Returns:
        True if a TCP connection to the URL's host:port succeeded.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as exc:
        logger.warning(f"{label} ({host}:{port}) not reachable ({exc}).")
        return False


def shared_stt_reachable(base_url: str = SHARED_STT_BASE_URL) -> bool:
    """Check the shared faster-whisper STT service."""
    return _tcp_reachable(base_url, "Shared STT service")


def qwen_omni_reachable(base_url: str = QWEN_OMNI_BASE_URL) -> bool:
    """Check the remote Qwen-Omni TTS server."""
    return _tcp_reachable(base_url, "Qwen-Omni TTS")


def shared_qwen_tts_reachable(base_url: str = SHARED_TTS_BASE_URL) -> bool:
    """Check the shared local Qwen3-TTS service."""
    return _tcp_reachable(base_url, "Shared Qwen-TTS service")


def make_stt(stt_cfg: dict):
    """Construct an STT plugin instance from an engine-config dict.

    Args:
        stt_cfg: ``{"engine": ..., "model": ...}``, same shape as
            ``system_config.py``'s ``stt`` section.

    Returns:
        A ``livekit.agents.stt.STT`` instance.
    """
    engine = stt_cfg.get("engine", "whisper_shared")
    model = stt_cfg.get("model", WHISPER_MODEL)

    if engine == "whisper_shared":
        if shared_stt_reachable():
            from plugins.shared_whisper_stt import SharedWhisperSTT

            logger.info(f"STT: shared faster-whisper service ({SHARED_STT_BASE_URL})")
            return SharedWhisperSTT(base_url=SHARED_STT_BASE_URL)
        logger.warning(
            "stt engine=whisper_shared but the shared service isn't "
            "reachable - falling back to in-process faster-whisper."
        )

    from plugins.whisper_stt import FasterWhisperSTT

    logger.info(f"STT: faster-whisper ({model})")
    return FasterWhisperSTT(model=model)


def make_tts(tts_cfg: dict):
    """Construct a TTS plugin instance from an engine-config dict.

    Args:
        tts_cfg: ``{"engine": ..., "model": ...}``, same shape as
            ``system_config.py``'s ``tts`` section.

    Returns:
        A ``livekit.agents.tts.TTS`` instance.
    """
    from livekit.plugins import openai

    engine = tts_cfg.get("engine", "qwen_omni")

    if engine == "qwen_shared":
        if shared_qwen_tts_reachable():
            from plugins.shared_qwen_tts import SharedQwenTTS

            logger.info(f"TTS: shared Qwen3-TTS service ({SHARED_TTS_BASE_URL})")
            return SharedQwenTTS(base_url=SHARED_TTS_BASE_URL)
        logger.warning(
            "tts engine=qwen_shared but the shared service isn't reachable "
            "- falling back to Qwen (remote)."
        )
    elif engine == "qwen_local_subprocess":
        from plugins.qwen_tts import QwenSubprocessTTS

        logger.info("TTS: Qwen (LOCAL subprocess, legacy fallback)")
        return QwenSubprocessTTS()

    logger.info(f"TTS: Qwen3-TTS via vLLM-Omni (remote, {QWEN_OMNI_BASE_URL})")
    return openai.TTS(
        model=QWEN_OMNI_MODEL,
        voice=tts_cfg.get("voice", QWEN_OMNI_VOICE),
        base_url=QWEN_OMNI_BASE_URL,
        api_key="not-needed",
        response_format="wav",
    )


async def warm_up_vllm(served_model_name: str) -> None:
    """One tiny completion so the first real turn doesn't eat vLLM's
    cold-start spike.

    Args:
        served_model_name: The model name vLLM is currently serving.
    """
    try:
        async with aiohttp.ClientSession() as http, http.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": served_model_name,
                "max_tokens": 4,
                "messages": [{"role": "user", "content": "ping"}],
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            body = await resp.text()
            if resp.status != 200:
                logger.error(
                    f"vLLM warm-up ping got HTTP {resp.status} - "
                    f"served_model_name={served_model_name!r} likely "
                    f"doesn't match what vLLM is serving. Body: {body[:300]}"
                )
                return
        logger.info("vLLM warm-up ping OK.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"vLLM warm-up ping failed (continuing): {exc}")


async def warm_up_qwen_omni() -> None:
    """One tiny synthesis request so the first real caller doesn't eat
    PC2's cold-start cost. Only meaningful if the reachability check
    already found PC2 up."""
    try:
        async with aiohttp.ClientSession() as http, http.post(
            f"{QWEN_OMNI_BASE_URL}/audio/speech",
            json={
                "model": QWEN_OMNI_MODEL,
                "input": "warm up",
                "voice": QWEN_OMNI_VOICE,
                "response_format": "wav",
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(
                    f"Qwen-Omni warm-up got HTTP {resp.status} from "
                    f"{QWEN_OMNI_BASE_URL}. Body: {body[:300]}"
                )
                return
            await resp.read()
        logger.info(f"Qwen-Omni (PC2, {QWEN_OMNI_BASE_URL}) warm-up OK.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Qwen-Omni warm-up ping failed (continuing): {exc}")
