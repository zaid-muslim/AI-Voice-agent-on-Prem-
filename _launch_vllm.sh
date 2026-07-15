#!/usr/bin/env bash
cd "/home/nauyan/Desktop/Voice agents/Pipeline"
export CUDA_HOME=/home/nauyan/miniconda3/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:/home/nauyan/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
# Skip flashinfer's JIT-compiled sampler (its bundled CCCL headers conflict with the
# toolkit headers on this box) -- falls back to vLLM's built-in sampler, no JIT needed.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec /home/nauyan/miniconda3/bin/vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --served-model-name qwen2.5-14b-awq \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.62 --max-model-len 8192 \
  --port 8000
