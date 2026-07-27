# Shared image for the LiveKit agent worker + token/admin/dev FastAPI
# servers (app/main.py, token_server.py, admin_server.py, dev_server.py) -
# same code, same deps, different CMD per docker-compose service. Needs
# CUDA/cuDNN because the in-process faster-whisper STT fallback
# (plugins/whisper_stt.py) and Silero VAD run on this image's GPU even
# though the DEFAULT STT path talks to the separate stt-service container.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /srv/app

COPY app/requirements.txt .
RUN pip3 install -r requirements.txt

COPY app/ .

# Downloads livekit-agents' VAD/turn-detector model files at BUILD time so
# containers start warm (mirrors main.py's own "download-files" step,
# normally run once by hand before `python main.py start`). Needs network
# access during `docker build`; if unavailable, this is a no-op and the
# same download happens lazily on first container start instead.
RUN python3 main.py download-files || true

EXPOSE 7860 7870 7871
CMD ["python3", "main.py", "start"]
