# PC2 ("TTS box") image: serves Qwen3-TTS via vLLM-Omni, one persistent
# GPU-warm process every PC1 agent worker/room hits over the network for
# TTS - see the root README's architecture section for the two-machine
# split this is half of.
#
# Deliberately NOT vendoring vllm-project/vllm-omni's source into this repo
# (it's a large third-party project with its own git history) - installed
# straight from its own repository instead, same as the technical setup
# notes this project already followed by hand.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN pip3 install git+https://github.com/vllm-project/vllm-omni.git

EXPOSE 8091
ENTRYPOINT ["vllm", "serve"]
# docker-compose.pc2.yml's `command:` supplies the model path + --omni
# --port flags - see that file for the actual invocation.
CMD ["--omni", "--port", "8091"]
