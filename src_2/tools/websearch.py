import asyncio
from datetime import datetime
from urllib.parse import urlparse
from ddgs import DDGS
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams

MAX_RESULTS = 3
MAX_SNIPPET_CHARS = 200

# Hard cap on how long a search may hold the turn hostage. DDG throttling
# can stall requests for tens of seconds; past this, we bail gracefully so
# the model can answer from its own knowledge instead of leaving the user
# in dead air after the filler phrase.
SEARCH_TIMEOUT_SECS = 6.0

# Per-request network timeout inside DDGS itself (connect/read), kept below
# the overall cap so the library fails before we have to abandon the thread.
DDGS_HTTP_TIMEOUT_SECS = 5

# Recency filter passed to DDGS. "m" = past month; use "w" (week) or "d"
# (day) if the agent is mostly asked about breaking news. THIS - not
# appending a date string to the query - is how to bias toward fresh
# results: DDG treats appended dates as literal keywords to match, which
# actively degrades results (pages rarely contain today's ISO date).
DEFAULT_TIMELIMIT = "m"


async def web_search(params: FunctionCallParams, query: str):
    """Search the web for current information.

    Args:
        query: What to search for.
    """
    # Let the person hear something immediately instead of dead air while
    # the search runs. NOTE: verify `append_to_context` exists on
    # TTSSpeakFrame in your installed pipecat wheel; if it's not a field,
    # this raises TypeError on the first tool call - drop the kwarg then.
    await params.llm.push_frame(
        TTSSpeakFrame("Let me check that for you.", append_to_context=False)
    )

    try:
        # Overall deadline on top of DDGS's own HTTP timeout: to_thread
        # can't be force-killed, but wait_for lets US stop waiting and
        # return control to the turn; the orphaned thread just finishes
        # quietly in the background.
        results = await asyncio.wait_for(
            _run_search(query), timeout=SEARCH_TIMEOUT_SECS
        )
    except asyncio.TimeoutError:
        await params.result_callback(
            "Search timed out. Answer from your own knowledge and say the "
            "information could not be verified as current."
        )
        return
    except Exception as exc:
        await params.result_callback(f"Search failed: {exc}")
        return
    await params.result_callback(results)


async def _run_search(query: str, max_results: int = MAX_RESULTS) -> str:
    # DDGS().text() is synchronous/blocking (network + HTML parsing) -
    # run it off the event loop so it doesn't stall the pipeline.
    def _search():
        with DDGS(timeout=DDGS_HTTP_TIMEOUT_SECS) as ddgs:
            return ddgs.text(
                query,
                max_results=max_results,
                timelimit=DEFAULT_TIMELIMIT,
            )

    raw_results = await asyncio.to_thread(_search)
    if not raw_results:
        return f"No results found for '{query}'."

    lines = []
    for r in raw_results:
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "").strip()
        if len(body) > MAX_SNIPPET_CHARS:
            body = body[:MAX_SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."
        source = urlparse(href).netloc.removeprefix("www.") if href else ""
        label = f"{title} ({source})" if source else title
        lines.append(f"- {label}: {body}")

    # Date goes in the RESULT for the model to reason about staleness -
    # not in the query, where DDG would match it as literal keywords.
    today = datetime.now().strftime("%Y-%m-%d")
    return f"Search results (today is {today}):\n" + "\n".join(lines)
