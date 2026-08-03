"""Tool discovery and allow-list enforcement for a domain pack.

DECISION: hand-rolled registry, not MCP, for v1. Every tool this platform
needs today is a first-party Python callable running in-process against
local SQLite/RAG/JSON, with no external-service boundary to cross - MCP's
actual value (vendor-neutral, swappable tool *servers* callable across a
process/network boundary) buys nothing here, and only costs a JSON-RPC hop
plus the (currently uninstalled) ``mcp`` extra. Confirmed directly against
the installed ``livekit-agents==1.6.6`` that ``Agent.__init__`` already
accepts both ``tools=[...]`` and ``mcp_servers=[...]`` natively - so a
future domain pack can add an ``mcp_servers:`` block to its manifest and
have ``agent_base.py`` pass it straight through, with zero change to this
module or ``BaseDomainAgent`` itself. That's the documented scaling path
(e.g. a domain needing a third-party vendor's tool server), not built now.

Given a pack's ``tools.module`` + ``tools.allow_list``, this module
imports that module, collects every allow-listed name that resolves to an
``@function_tool``-decorated callable, and raises with EVERY missing/
misnamed tool listed at once - same "collect every problem" convention as
``system_config.save_config()``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from livekit.agents.llm import FunctionTool, RawFunctionTool


def load_tools(module_path: str, allow_list: tuple[str, ...]) -> list[Callable]:
    """Import a pack's tools module and collect its allow-listed tools.

    Args:
        module_path: Dotted import path (``DomainPack.tools.module``).
        allow_list: Tool names that must resolve to decorated callables
            in that module (``DomainPack.tools.allow_list``).

    Returns:
        The resolved callables, in ``allow_list`` order - passed straight
        into ``Agent(tools=...)``.

    Raises:
        ImportError: If ``module_path`` can't be imported.
        ValueError: If any allow-listed name is missing, or resolves to
            something that isn't an ``@function_tool``-decorated
            callable. Lists every problem found, not just the first.
    """
    module = importlib.import_module(module_path)
    problems: list[str] = []
    tools: list[Callable] = []
    for name in allow_list:
        obj = getattr(module, name, None)
        if obj is None:
            problems.append(f"{module_path}.{name} does not exist")
            continue
        if not isinstance(obj, (FunctionTool, RawFunctionTool)):
            problems.append(
                f"{module_path}.{name} exists but is not an "
                "@function_tool-decorated callable"
            )
            continue
        tools.append(obj)
    if problems:
        raise ValueError(
            f"Invalid tool allow_list for {module_path}:\n  - "
            + "\n  - ".join(problems)
        )
    return tools


def check_deterministic_backing(module_path: str, allow_list: tuple[str, ...]) -> list[str]:
    """Enforce ``CompliancePolicy.require_deterministic_tool_backing``.

    A pragmatic, name-based check (not brittle static analysis of
    function bodies): a compliant pack's tools module must export a
    module-level ``DETERMINISTIC: set[str]`` naming every allow-listed
    tool, asserting each one is backed by real I/O (a DB/RAG call) rather
    than something the LLM could assert on its own.

    Args:
        module_path: Dotted import path of the pack's tools module.
        allow_list: The pack's tool allow-list.

    Returns:
        A list of problems (missing ``DETERMINISTIC`` export, or tools
        present in ``allow_list`` but absent from ``DETERMINISTIC``) -
        empty if the module fully complies.
    """
    module = importlib.import_module(module_path)
    deterministic = getattr(module, "DETERMINISTIC", None)
    if deterministic is None:
        message = (
            f"{module_path} has no module-level DETERMINISTIC set[str], "
            "required because this pack's compliance profile sets "
            "require_deterministic_tool_backing=True"
        )
        return [message]
    missing = [name for name in allow_list if name not in deterministic]
    if missing:
        return [f"{module_path}.DETERMINISTIC is missing allow-listed tool(s): {missing}"]
    return []
