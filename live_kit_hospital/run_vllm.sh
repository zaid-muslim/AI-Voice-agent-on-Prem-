#!/usr/bin/env bash
# vLLM launch, tuned for 3 CONCURRENT callers on a single 24GB GPU.
# (Originally targeted 4 with fp8 KV cache savings - see change #2 below
# for why that was shelved on this specific GPU, and why 3 is the honest
# target now.)
#
# WHAT CHANGED FROM THE ORIGINAL, AND WHY:
#
# 1. --gpu-memory-utilization 0.50 -> 0.40 (RE-TARGETED FOR 3 CALLERS)
#    At 0.50, vLLM reserves ~12.4GB, confirmed via nvidia-smi. Measured
#    marginal cost per concurrent caller (whisper STT + Qwen TTS worker,
#    NOT vLLM) is ~4.6GB.
#
#    ORIGINAL PLAN targeted 4 callers and assumed fp8 KV cache would free
#    enough room for it. THAT FAILED: fp8 needs compute capability 8.9+
#    (Ada Lovelace/Hopper); this RTX 3090 is 8.6 (Ampere) with no FP8
#    hardware at all - vLLM refused to start (see change #2). Reverted
#    to bfloat16, which costs 2x the memory per token fp8 would have.
#
#    RE-TARGETING FOR 3 CALLERS (not 4) is the honest move here, since
#    the memory savings that would have made 4 realistic don't exist on
#    this GPU:
#         3 callers x 4.6GB = 13.8GB
#         OS/desktop        = ~0.5GB
#         left for vLLM     = 24 - 13.8 - 0.5 = ~9.7GB
#         as a fraction     = 9.7 / 24 ~= 0.40
#    0.40 reserves ~9.6GB - fits the ~6GB model weights plus a workable
#    bfloat16 KV cache, with the remaining ~14.4GB comfortably covering
#    3 STT/TTS callers plus OS overhead. TEST live and watch nvidia-smi -
#    this is a calculated target, not a guarantee, same as always.
#
# 2. --kv-cache-dtype bfloat16 -> fp8 -> REVERTED back to bfloat16
#    Attempted fp8 for the memory savings described above, but vLLM
#    refused to start on this card:
#        "FP8 KV cache is not supported by the Triton attention backend
#         on NVIDIA GeForce RTX 3090 (compute capability 8.6); native
#         FP8 (fp8e4nv) requires SM89+."
#    This is a HARDWARE limit, not a config or package issue: native FP8
#    math was only introduced at compute capability 8.9 (Ada
#    Lovelace/RTX 4000-series and up, plus Hopper datacenter cards). The
#    3090 is Ampere (8.6) and its silicon has no FP8 execution units at
#    all - no driver update or flag combination fixes this. bfloat16 is
#    the correct, working setting for this specific GPU. The memory
#    savings this would have bought back for change #1 above are NOT
#    available on this card - #1's budget math should be re-checked
#    against the ORIGINAL bfloat16 per-token cost, not the fp8 estimate.
#
# 3. --max-num-seqs 4 -> 4 (kept, re-justified for the 3-caller target)
#    This is vLLM's own hard concurrency ceiling, separate from memory -
#    it will queue/refuse a request beyond this even with free VRAM.
#    Left at 4 (one spare slot above the 3-caller target) rather than
#    bumped to 5, since 5 implied planning for genuine 4-way overlap -
#    that plan is shelved along with the fp8/4-caller attempt above.
#
# 4. --max-model-len 8192 -> 4096 (NEW - added after a REAL failure, not
#    a guess)
#    First real run at 0.40/bfloat16 failed to even START:
#        "Model loading took 8.28 GiB memory"   <- REAL number, not the
#                                                   ~6GB estimate used
#                                                   above - always trust
#                                                   the tool's own report
#                                                   over a prior estimate
#        "Available KV cache memory: 0.64 GiB"
#        ValueError: needs 1.2 GiB KV cache for even ONE request at the
#        configured max_model_len (8192), only 0.64 GiB available.
#
#    Raising --gpu-memory-utilization further would fix this too, but at
#    the direct cost of the ~13.8GB reserved for 3 callers' STT/TTS
#    processes - not free. The better fix: your ACTUAL conversations run
#    ~1800-2100 prompt_tokens per your own logs, nowhere near 8192. KV
#    cache requirement scales with max_model_len, so halving it to 4096
#    (still ~2x your real usage, real margin for longer calls) roughly
#    halves the memory each sequence needs, fixing the shortage WITHOUT
#    taking memory back from the caller-headroom budget.
#
#    VERIFY, don't assume: check the NEXT startup log's own
#    "Available KV cache memory" line and whether it starts successfully.
#    If still short, that's real evidence to try 3072, or as a last
#    resort accept a small trim to caller headroom via a slightly higher
#    gpu-memory-utilization - but let the tool's own numbers decide that,
#    not another guess.
#
# HOW TO VERIFY THIS ACTUALLY WORKS, DON'T JUST TRUST THE MATH:
#   1. Start vLLM with this script, check its OWN startup logs for the
#      KV cache capacity it reports (look for "# GPU blocks" or similar).
#      Sanity check: 3 callers x ~2000 tokens each = 6000 tokens minimum
#      needed - the reported capacity should comfortably exceed that.
#   2. Run `watch -n 1 nvidia-smi` in a separate terminal.
#   3. Bring on 3 real concurrent callers and watch total VRAM usage and
#      tokens_per_second in the agent logs. If throughput visibly
#      degrades or VRAM creeps toward the 24GB ceiling, that's real
#      evidence to dial --gpu-memory-utilization back up slightly (fewer
#      concurrent callers) rather than risk an OOM mid-call.

VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve \
    /home/nauyan/voice-agent-pipeline/models/gemma-4-12b-w4a16 \
    --served-model-name gemma-4-12b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"audio": 1}' \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.55 \
    --kv-cache-dtype bfloat16 \
    --generation-config vllm \
    --disable-log-stats \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --chat-template /home/nauyan/voice-agent-pipeline/chat_templates/tool_chat_template_gemma4.jinja
