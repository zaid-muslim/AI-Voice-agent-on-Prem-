# Shared local Qwen3-TTS 0.6B service - see tts_service/server.py's
# docstring for why this is its own container: ONE persistent, GPU-warm
# model shared by every agent worker/room, ~2-3x faster to first audio
# than the default remote PC2 path (measured, see README §3/§4), at the
# cost of THIS machine's GPU budget instead of PC2's. Opt-in - see
# docker-compose.yml's "local-tts" profile note.
#
# PYTHON 3.10, not 3.12 (unlike app.Dockerfile/stt.Dockerfile): this
# matches the exact interpreter/wheel versions actually verified working
# on this project's dev machine (.venv-voice, torch==2.6.0+cu124,
# faster-qwen3-tts==0.3.0 - see tts_service/requirements.txt's header
# comment). Ubuntu 22.04 ships Python 3.10 as its system python3, so no
# deadsnakes PPA is needed here the way stt.Dockerfile needs one for 3.12.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/tts

COPY tts_service/requirements.txt .
# Legacy build-time prerequisites FIRST, own step: faster-qwen3-tts pulls
# in `sox` transitively, and sox's legacy setup.py does plain `import
# numpy` / `import typing_extensions` at BUILD time (not declared as a
# proper PEP 517 build-system requirement) - a single
# `pip install -r requirements.txt` doesn't guarantee either is actually
# installed yet when pip's resolver reaches sox's metadata generation.
# Confirmed live, twice: first failure was numpy missing, fixing that
# surfaced typing_extensions missing next - both are exactly this same
# "legacy setup.py assumes its build-time imports are already present"
# problem, so install the small, common set of packages this class of
# issue needs up front rather than whack-a-mole one broken module at a
# time.
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install "numpy>=1.24" "typing_extensions>=4.10"
RUN python3 -m pip install -r requirements.txt

COPY tts_service/server.py .

ENV QWEN_MODEL_ID=/models/Qwen3-TTS-0.6B-custom \
    QWEN_TTS_DTYPE=bfloat16 \
    QWEN_SPEAKER=aiden \
    QWEN_LANGUAGE=English \
    QWEN_CHUNK_SIZE=8 \
    QWEN_TTS_PORT=8021

EXPOSE 8021
# Model weights are volume-mounted (see docker-compose.yml), same pattern
# as vllm's /models mount - not baked into the image.
CMD ["python3", "server.py"]
