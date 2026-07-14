"""
Adaptive filler wrapper for tool calls.

Wraps any async tool function (RAG lookup, availability check, booking call,
web search, anything) so that a filler phrase is only spoken if the call is
still running past a time threshold - never on fast calls, always on slow
ones. This replaces the old pattern of an unconditional "let me check that
for you" line (which feels robotic on a 100ms lookup) with a race: the tool
call and a timer start together, and whichever finishes first decides what
the caller hears next.

    - Tool finishes before threshold  -> no filler is ever spoken.
    - Tool still running at threshold -> filler is spoken, THEN we keep
      waiting (the tool call is never interrupted or restarted).

USAGE - drop-in decorator on any Pipecat-style tool function:

    @with_adaptive_filler(threshold_secs=0.7, filler_text="One moment, let me check that.")
    async def check_availability(params: FunctionCallParams, department: str, date: str):
        ...
        await params.result_callback(result)

The decorated function's own error handling is untouched - if it catches its
own exceptions and reports them via result_callback (as web_search.py does),
that continues to work exactly the same with or without this wrapper.

NOTE ON EXISTING web_search.py: it currently pushes its filler line
unconditionally, before the search even starts. Once this wrapper is
adopted, remove that hardcoded line and wrap web_search with
@with_adaptive_filler instead - otherwise fast searches get an unnecessary
filler AND this one, doubling up.

VERIFY BEFORE PRODUCTION USE: `TTSSpeakFrame(text, append_to_context=False)`
is the same API surface flagged in websearch.py - confirm `append_to_context`
exists on your installed pipecat's TTSSpeakFrame; if not, drop the kwarg in
_default_filler_frame() below.
"""

import asyncio
import functools
from typing import Callable, Optional

try:
    from loguru import logger
except ImportError:  # loguru is a normal dependency in the target project;
    # this fallback only exists so this file's self-test can run in
    # environments without it installed.
    import logging

    logger = logging.getLogger(__name__)


def _default_filler_frame(text: str):
    # Lazy import: this module has no hard pipecat dependency until a filler
    # actually needs to be spoken, which also keeps it testable without
    # pipecat installed (see self-test at the bottom).
    from pipecat.frames.frames import TTSSpeakFrame

    return TTSSpeakFrame(text, append_to_context=False)


def with_adaptive_filler(
    threshold_secs: float = 0.7,
    filler_text: str = "One moment, let me check that.",
    make_filler_frame: Callable[[str], object] = _default_filler_frame,
):
    """
    threshold_secs: how long the tool call gets before we speak up. Tune this
        against real measured latencies for each tool once they're wired to
        real backends - 0.7s is a placeholder starting point, not a measured
        value.
    filler_text: what to say if the threshold is crossed. Keep it short and
        generic enough to fit any tool ("One moment, let me check that.")
        unless you want per-tool phrasing.
    make_filler_frame: override only for testing (see self-test below) or if
        you want something other than a plain TTSSpeakFrame.
    """

    def decorator(tool_fn: Callable):
        @functools.wraps(tool_fn)
        async def wrapper(params, *args, **kwargs):
            task = asyncio.ensure_future(tool_fn(params, *args, **kwargs))
            try:
                # shield: prevents wait_for's timeout from cancelling the
                # underlying call. On timeout, the task keeps running in the
                # background - we're only giving up on WAITING for it here,
                # not stopping it.
                await asyncio.wait_for(asyncio.shield(task), timeout=threshold_secs)
            except asyncio.TimeoutError:
                logger.debug(
                    f"{tool_fn.__name__}: crossed {threshold_secs}s threshold, "
                    f"speaking filler and continuing to wait"
                )
                await params.llm.push_frame(make_filler_frame(filler_text))
                await task  # now actually wait for it to finish

        return wrapper

    return decorator


# --- self-test: verify the race behaves correctly before wiring this in ----

if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    # Lightweight fakes so this test needs no pipecat install at all - the
    # decorator's default path still uses real pipecat frames in production
    # (see _default_filler_frame's lazy import above).
    def _fake_filler_frame(text):
        return SimpleNamespace(text=text)

    class FakeLLM:
        def __init__(self):
            self.pushed = []

        async def push_frame(self, frame):
            self.pushed.append(frame)

    class FakeParams:
        def __init__(self):
            self.llm = FakeLLM()
            self.result = None

        async def result_callback(self, result):
            self.result = result

    @with_adaptive_filler(
        threshold_secs=0.4,
        filler_text="One moment.",
        make_filler_frame=_fake_filler_frame,
    )
    async def fast_tool(params, label):
        await asyncio.sleep(0.1)  # well under the 0.4s threshold
        await params.result_callback(f"fast result: {label}")

    @with_adaptive_filler(
        threshold_secs=0.4,
        filler_text="One moment.",
        make_filler_frame=_fake_filler_frame,
    )
    async def slow_tool(params, label):
        await asyncio.sleep(1.0)  # crosses the 0.4s threshold
        await params.result_callback(f"slow result: {label}")

    async def _run():
        results = []

        p1 = FakeParams()
        t0 = time.perf_counter()
        await fast_tool(p1, "availability check")
        dt1 = time.perf_counter() - t0
        ok1 = p1.result == "fast result: availability check" and len(p1.llm.pushed) == 0
        results.append(("fast tool: no filler spoken, correct result", ok1))

        p2 = FakeParams()
        t0 = time.perf_counter()
        await slow_tool(p2, "booking call")
        dt2 = time.perf_counter() - t0
        ok2 = (
            p2.result == "slow result: booking call"
            and len(p2.llm.pushed) == 1
            and p2.llm.pushed[0].text == "One moment."
            and 0.9 < dt2 < 1.3
        )  # completed around the real 1.0s sleep,
        # not cut short at the 0.4s threshold
        results.append(
            (
                "slow tool: filler spoken once, correct result, "
                "waited for real completion",
                ok2,
            )
        )

        for desc, ok in results:
            print(f"[{'PASS' if ok else 'FAIL'}] {desc}")
        print(
            f"\nfast tool took {dt1:.2f}s, slow tool took {dt2:.2f}s "
            f"(filler fired at 0.4s, real result arrived at ~1.0s)"
        )

        all_ok = all(ok for _, ok in results)
        if not all_ok:
            print("DO NOT wire this into the pipeline until all cases pass.")
        return all_ok

    passed = asyncio.run(_run())
    exit(0 if passed else 1)
