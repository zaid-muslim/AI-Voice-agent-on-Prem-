"""
Gemma-4-E4B native audio brain -- STREAMING + MEMORY + SEARCH version.
Runs in venv-brain.
"""

import re
import threading
from collections import deque
from typing import Iterator, Optional

import torch

from audio_utils import clip_to_max_audio_len
from hospital_agent.src.web_search import search_web

# ---- CONFIG: change this to your actual absolute path ---
MODEL_PATH = "/home/nauyan/voice-agent-pipeline/models/gemma-4-E4B-it"

MAX_NEW_TOKENS = 640
MAX_MEMORY_TURNS = 60  # bounds prompt growth on long sessions

# Google's documented standardized sampling config for Gemma 4 (all sizes).
SAMPLING_KWARGS = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)

# SOTA FIX: Updated prompt to mandate a verbal acknowledgment while searching.
SYSTEM_PROMPT = (
    "You are a warm, emotionally present voice companion. "
    "If you genuinely need current information (news, weather, facts), you can request a web search.\n\n"
    "Always output exactly this structure:\n"
    "SEARCH: a short search query, ONLY if needed. Otherwise write exactly NONE\n"
    "REPLY: If SEARCH is NONE, write your full response here. If SEARCH is NOT NONE, write a very short phrase here like 'Let me check that.' or 'Give me a second.'\n"
    "LOG: a short sentence summarizing the user's intent.\n\n"
    "Never use markdown. Output ONLY these three lines."
)

print(f"⏳ Loading offline multimodal brain from {MODEL_PATH}...")

from transformers import (  # noqa: E402
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = getattr(processor, "tokenizer", processor)

model = None
_load_errors = []
for _cls_name in (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForCausalLM",
):
    try:
        import transformers as _tf

        _cls = getattr(_tf, _cls_name)
        model = _cls.from_pretrained(
            MODEL_PATH,
            device_map="cuda",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        print(f"✅ Loaded with {_cls_name}")
        break
    except Exception as e:  # noqa: BLE001
        _load_errors.append(f"{_cls_name}: {e}")
        continue

if model is None:
    raise RuntimeError(
        "Could not load Gemma-4-E4B with any known multimodal class. Errors:\n"
        + "\n".join(_load_errors)
    )

print("✅ Multimodal Brain Ready.\n")

_memory: deque[tuple[str, str]] = deque(maxlen=MAX_MEMORY_TURNS)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def reset_memory() -> None:
    """Call this to start a fresh conversation (e.g. on a wake-word reset)."""
    _memory.clear()


def _field(text: str, label: str, next_label: str, start: int = 0) -> str:
    i = text.find(label, start)
    if i == -1:
        return ""
    i += len(label)
    j = text.find(next_label, i)
    if j == -1:
        j = len(text)

    # SOTA FIX: Strip common LLM bleed-through characters
    return text[i:j].replace("REPLY:", "").replace("SEARCH:", "").strip()


def _build_messages(audio_array) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for log_text, reply_text in _memory:
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": log_text}]}
        )
        messages.append({"role": "assistant", "content": reply_text})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Listen to the following audio and respond."},
                {"type": "audio", "audio": audio_array},
            ],
        }
    )
    return messages


class _CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancel_event: threading.Event) -> None:
        self._cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: D401, ANN001
        return self._cancel_event.is_set()


def _generate_stream(
    messages: list, cancel_event: Optional[threading.Event] = None
) -> Iterator[str]:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        **SAMPLING_KWARGS,
    )
    if cancel_event is not None:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_CancelStoppingCriteria(cancel_event)]
        )

    thread = threading.Thread(
        target=model.generate, kwargs=generate_kwargs, daemon=True
    )
    thread.start()
    for delta in streamer:
        yield delta
    thread.join()


def _run_pass(
    messages: list, cancel_event: Optional[threading.Event] = None
) -> Iterator[str]:
    full_text = ""
    reply_start: Optional[int] = None
    reply_end: Optional[int] = None
    emitted_upto = 0
    search_text = log_text = ""

    for delta in _generate_stream(messages, cancel_event):
        full_text += delta

        if reply_start is None:
            idx = full_text.find("REPLY:")
            if idx == -1:
                continue
            reply_start = idx + len("REPLY:")
            emitted_upto = reply_start
            search_text = _field(full_text, "SEARCH:", "REPLY:")

        if reply_end is None:
            log_idx = full_text.find("LOG:", reply_start)
            if log_idx != -1:
                reply_end = log_idx

        boundary = reply_end if reply_end is not None else len(full_text)
        unsent = full_text[emitted_upto:boundary]
        while True:
            m = _SENTENCE_END.search(unsent)
            if not m:
                break
            cut = m.end()
            sentence, unsent = unsent[:cut], unsent[cut:]

            # SOTA FIX: Aggressively strip LLM formatting bleed
            clean_sentence = (
                sentence.replace("SEARCH: NONE", "")
                .replace("SEARCH:", "")
                .replace("REPLY:", "")
                .strip()
            )
            if clean_sentence:
                yield clean_sentence
            emitted_upto += cut

    if reply_start is not None:
        boundary = reply_end if reply_end is not None else len(full_text)
        tail = full_text[emitted_upto:boundary].strip()

        clean_tail = (
            tail.replace("SEARCH: NONE", "")
            .replace("SEARCH:", "")
            .replace("REPLY:", "")
            .strip()
        )
        if clean_tail:
            yield clean_tail
        reply_text = full_text[reply_start:boundary].strip()
        if reply_end is not None:
            log_text = full_text[reply_end + len("LOG:") :].strip()
    else:
        raw = full_text.strip()
        clean_raw = (
            raw.replace("SEARCH: NONE", "")
            .replace("SEARCH:", "")
            .replace("REPLY:", "")
            .strip()
        )
        if clean_raw:
            yield clean_raw
        reply_text = raw

    return log_text, search_text, reply_text


def respond_to_audio(
    audio_array,
    sampling_rate: int = 16000,
    cancel_event: Optional[threading.Event] = None,
) -> Iterator[dict]:
    if sampling_rate != 16000:
        raise ValueError(
            "respond_to_audio expects 16kHz audio; resample before calling "
            "(see audio_utils.resample_to_16k)."
        )
    audio_array = clip_to_max_audio_len(audio_array, sampling_rate)

    messages = _build_messages(audio_array)
    gen = _run_pass(messages, cancel_event)

    log_text = search_query = reply_text = ""
    while True:
        try:
            sentence = next(gen)
        except StopIteration as stop:
            log_text, search_query, reply_text = stop.value
            break
        if cancel_event is not None and cancel_event.is_set():
            continue
        yield {"type": "sentence", "text": sentence}

    was_cancelled = cancel_event is not None and cancel_event.is_set()

    if not was_cancelled and search_query and search_query.strip().upper() != "NONE":
        print(f"🔎 Searching: {search_query}")
        results = search_web(search_query)

        followup = messages[:-1] + [
            {"role": "user", "content": [{"type": "text", "text": log_text}]},
            {"role": "assistant", "content": f"SEARCH: {search_query}"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Search results:\n{results}\n\n"
                            "The user can't see these results. Using them, write REPLY "
                            "only (skip SEARCH/LOG) in the same spoken style."
                        ),
                    }
                ],
            },
        ]

        full_text = ""
        emitted_upto = 0
        stop_idx: Optional[int] = None
        sentences: list[str] = []

        for delta in _generate_stream(followup, cancel_event):
            full_text += delta

            if stop_idx is None:
                li = full_text.find("LOG:")
                if li != -1:
                    stop_idx = li
            boundary = stop_idx if stop_idx is not None else len(full_text)
            unsent = full_text[emitted_upto:boundary]
            while True:
                m = _SENTENCE_END.search(unsent)
                if not m:
                    break
                cut = m.end()
                sentence, unsent = unsent[:cut], unsent[cut:]

                clean_sentence = (
                    sentence.replace("SEARCH: NONE", "")
                    .replace("SEARCH:", "")
                    .replace("REPLY:", "")
                    .strip()
                )
                if clean_sentence:
                    sentences.append(clean_sentence)
                    if cancel_event is None or not cancel_event.is_set():
                        yield {"type": "sentence", "text": clean_sentence}
                emitted_upto += cut

        boundary = stop_idx if stop_idx is not None else len(full_text)
        tail = full_text[emitted_upto:boundary].strip()

        clean_tail = (
            tail.replace("SEARCH: NONE", "")
            .replace("SEARCH:", "")
            .replace("REPLY:", "")
            .strip()
        )
        if clean_tail:
            sentences.append(clean_tail)
            if cancel_event is None or not cancel_event.is_set():
                yield {"type": "sentence", "text": clean_tail}

        reply_text = " ".join(sentences)

    was_cancelled = cancel_event is not None and cancel_event.is_set()
    if log_text and not was_cancelled:
        _memory.append((log_text, reply_text))
    yield {"type": "done"}


if __name__ == "__main__":
    import librosa

    test_file = "data/audio_logs/test.wav"
    try:
        audio_array, _ = librosa.load(test_file, sr=16000)
        for event in respond_to_audio(audio_array, sampling_rate=16000):
            if event["type"] == "sentence":
                print(f"🤖 {event['text']}")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error: {e}")
