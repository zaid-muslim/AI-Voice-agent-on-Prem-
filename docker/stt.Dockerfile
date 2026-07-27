# Shared faster-whisper STT service - see stt_service/server.py's docstring
# for why this is its own container instead of code inside app.Dockerfile:
# ONE persistent, GPU-warm model shared by every agent worker/room, so
# concurrent callers don't each get their own loaded whisper model.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /srv/stt

COPY stt_service/requirements.txt .
RUN pip3 install -r requirements.txt

COPY stt_service/server.py .

ENV STT_MODEL=distil-large-v3 \
    STT_DEVICE=cuda \
    STT_COMPUTE_TYPE=int8_float16 \
    STT_NUM_WORKERS=4 \
    STT_PORT=8020

EXPOSE 8020
# The model auto-downloads from Hugging Face into HF_HOME on first start -
# mount a volume there (see docker-compose.yml's hf-cache volume) so a
# container restart doesn't re-download it every time.
CMD ["python3", "server.py"]
