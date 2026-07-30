# Shared image for the direct-audio pipeline's LiveKit worker
# (direct_audio_agent/agent.py) + its call server
# (direct_audio_agent/call_server.py) - same code, same deps, different
# CMD per docker-compose service, exactly mirroring app.Dockerfile's
# pattern for the existing agent/token-server/admin-server/dev-server
# quartet.
#
# WHY THIS IMAGE CONTAINS BOTH app/ AND direct_audio_agent/, NOT JUST
# THE LATTER: direct_audio_agent/agent.py and call_server.py both import
# app/main.py's RiversideReceptionist, prompts, hospital tools, and
# helpers directly (read-only) via a relative sys.path insert
# (`Path(__file__).resolve().parent.parent / "app"`) - see
# direct_audio_agent/README.md for why (reuse, not duplicate, the
# persona/tools/safety-gate). Preserving that same `app/` and
# `direct_audio_agent/` sibling layout inside the image (both under
# /srv) is what makes that path resolution work unchanged, identical to
# how it works when run bare-metal from the repo root.
#
# Uses the SAME app/requirements.txt as app.Dockerfile - direct_audio_agent
# adds no new runtime dependencies (aiohttp, numpy, livekit-agents,
# fastapi/uvicorn, python-dotenv, loguru are all already there).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python3.12 -m ensurepip --upgrade

WORKDIR /srv

COPY app/requirements.txt app/requirements.txt
RUN python3 -m pip install -r app/requirements.txt

COPY app/ app/
COPY direct_audio_agent/ direct_audio_agent/

WORKDIR /srv/direct_audio_agent

# Same "download VAD/turn-detector files at build time so the container
# starts warm" step as app.Dockerfile - a SEPARATE image needs its own
# copy, it isn't shared with the app image's layers.
RUN python3 agent.py download-files || true

EXPOSE 7862
CMD ["python3", "agent.py", "start"]
