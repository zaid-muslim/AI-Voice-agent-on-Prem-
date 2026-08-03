# Shared image for every domain_agent_core worker + call server
# (worker.py, call_server.py) - same code, same deps, one image serves
# EVERY domain pack (hospital, banking, ...); which pack an actual
# container runs is selected entirely by the DOMAIN_PACK environment
# variable at container start, per docker-compose.domains.yml's
# per-domain services.
#
# WHY THIS IMAGE CONTAINS app/, NOT JUST domain_agent_core/: the hospital
# pack's tools.py/safety_policy.py (Phase 0's regression baseline) import
# app/compat.py -> app/hospital_core/* directly (read-only) via the same
# sys.path-insert pattern direct-audio.Dockerfile already established for
# direct_audio_agent/ - see that Dockerfile's own comment for why this
# sibling-directory layout matters. Preserving app/ and domain_agent_core/
# as siblings under /srv keeps that path resolution identical to running
# bare-metal from the repo root.
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
COPY domain_agent_core/requirements.txt domain_agent_core/requirements.txt
RUN python3 -m pip install -r app/requirements.txt -r domain_agent_core/requirements.txt

COPY app/ app/
COPY domain_agent_core/ domain_agent_core/

WORKDIR /srv/domain_agent_core

# Same "download VAD/turn-detector model files at build time so the
# container starts warm" step as app.Dockerfile/direct-audio.Dockerfile -
# needs DOMAIN_PACK set even at build time since worker.py fails closed
# without it; the actual pack choice doesn't matter for this download
# step (the files are model-level, not domain-level).
RUN DOMAIN_PACK=hospital python3 worker.py download-files || true

CMD ["python3", "worker.py", "start"]
