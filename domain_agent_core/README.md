# domain_agent_core

A domain-configurable voice-agent platform: the exact same runtime becomes
a "hospital receptionist," a "banking agent," or any other vertical purely
by pointing it at a different **domain pack** - no forking code per
domain. Built alongside (and never modifying) this repo's existing
hospital voice pipeline (`app/`, `direct_audio_agent/`); see this
project's own `README.md` for that pipeline's documentation.

## Why this exists

`app/main.py` hardcodes one persona (`RiversideReceptionist`): hospital
tools, hospital safety-gate keywords, hospital filler phrases, a hospital
knowledge base. `domain_agent_core` generalizes that into a **core-plus-
domain** architecture - a stable, reusable runtime (`core/`) plus
swappable **domain packs** (`packs/<domain>/`), each a YAML manifest plus
its own tools/safety-policy/persona/knowledge-base.

## Architecture

```
domain_agent_core/
├── core/            # domain-agnostic runtime - never touched per domain
├── packs/
│   ├── hospital/     # ports RiversideReceptionist onto this runtime (regression baseline)
│   └── banking/       # a genuinely different vertical - the core thesis's proof point
├── worker.py          # ONE entrypoint for every domain, selected by DOMAIN_PACK
├── call_server.py     # per-domain browser call server (explicit LiveKit dispatch)
├── tests/
└── docker/
```

**One worker script, one `Agent` subclass** (`core/agent_base.
BaseDomainAgent`). A "hospital agent" and a "banking agent" are the same
running code, differing only in which `manifest.yaml` gets loaded at
process start.

### Core (`core/`) - reusable by every domain, unchanged

| Module | What it does |
|---|---|
| `domain_pack.py` | Typed manifest schema (`DomainPack` + friends), stdlib dataclasses + hand-written validation. |
| `domain_loader.py` | `AgentAssembler` - loads a manifest, validates it, resolves tools/safety/RAG/compliance into an `AssembledDomain`. |
| `agent_base.py` | `BaseDomainAgent(Agent)` - the one Agent subclass every pack shares. |
| `prompt_builder.py` | Jinja2 persona-template rendering, with the domain's own real local date and compliance clause injected. |
| `engine_factory.py` | STT/TTS backend construction + warm-ups (ported unchanged from `app/main.py` - already domain-agnostic). |
| `engine_config.py` | Per-pack llm/stt/tts engine selection store (same schema-v2 pattern as `app/system_config.py`). |
| `turn_filler.py` | Turn-level + per-tool filler mechanism (ported from `app/main.py`/`app/helpers.py`, phrases now per-pack). |
| `chat_ctx_guard.py` | Chat-context truncation guard (bounds prompt growth against the LLM's context window). |
| `thinking_filter.py` | Strips a leaked Gemma-4 `<\|channel>thought...` block from streamed text. |
| `safety_gate.py` | Generic two-layer safety-gate engine (regex match + `StopResponse` bypass); each pack supplies its own `SafetyPolicy`. |
| `rag_engine.py` | Semantic-search engine (sentence-transformers + keyword fallback), one instance per domain's own KB ("Silo" pattern). |
| `kb_store.py` | Atomic JSON load/save for a domain's knowledge base. |
| `tool_registry.py` | Resolves a pack's tool allow-list into callables; enforces deterministic-tool-backing for compliance profiles that require it. |
| `audit_log.py` | Compliance-aware, append-only, never-rotated audit logging. |
| `compliance_profiles.py` | `hipaa` / `pci_dss_glba` / `none` - concrete, enforced knobs, not just prompt text. |

### Domain packs (`packs/<domain>/`)

Each pack is a `manifest.yaml` plus:
- `persona.jinja2` - the system-prompt template.
- `tools.py` - `@function_tool`-decorated free functions, plus a
  `DETERMINISTIC: set[str]` if the pack's compliance profile requires it.
- `safety_policy.py` - a module-level `POLICY: SafetyPolicy`.
- `kb/<domain_id>_kb.json` - knowledge-base entries (optional).

See `packs/hospital/manifest.yaml` and `packs/banking/manifest.yaml` for
two working, tested examples of a full manifest.

## Running it

**Bare-metal** (same venv `app/main.py` uses):
```bash
DOMAIN_PACK=hospital venv/bin/python domain_agent_core/worker.py console
DOMAIN_PACK=banking  venv/bin/python domain_agent_core/worker.py console

DOMAIN_PACK=hospital venv/bin/python domain_agent_core/call_server.py   # :7863
DOMAIN_PACK=banking  venv/bin/python domain_agent_core/call_server.py   # :7864
```

**Docker** (layers on top of the existing `docker/docker-compose.yml` -
same shared `livekit-server`/`vllm`):
```bash
docker compose -f docker/docker-compose.yml \
  -f domain_agent_core/docker/docker-compose.domains.yml \
  --profile hospital --profile banking up -d --build
```
Run one profile alone, or both together (same GPU-contention caveat the
existing compose file already documents for its own `cascade`/
`direct-audio` pair applies here too).

## Tests

```bash
venv/bin/python -m pytest domain_agent_core/tests/ -v
```

13 Phase-0/1/2 tests, all passing: hospital-pack regression (system
prompt string-identical to `app/prompts.py`'s output, safety-gate 15/15
parity with `hospital_core/safety.py`, real booking-layer reachability
through this pack's own import path), banking-pack assembly (5 tools, its
own safety rules, `pci_dss_glba` compliance wired to silent duress
escalation), and compliance enforcement (redaction, never-rotated audit
files, load-time failure on missing deterministic-tool declarations).

## Adding a new domain

1. `mkdir packs/<domain>/`
2. Write `manifest.yaml` (copy `packs/banking/manifest.yaml` as a
   template - it's the more illustrative example since hospital's Phase 0
   pack imports `app/hospital_core/*` directly).
3. Write `persona.jinja2`, `tools.py` (with `DETERMINISTIC` if your
   compliance profile requires it), `safety_policy.py` (a `POLICY:
   SafetyPolicy`), and optionally `kb/<domain>_kb.json`.
4. `DOMAIN_PACK=<domain> venv/bin/python domain_agent_core/worker.py console`

No changes to `core/`, `worker.py`, `app/`, or `direct_audio_agent/` are
ever required to add a domain.

## Honest limitations

- Only two packs exist today (hospital, banking) - both assemble and pass
  their tests, but neither has been exercised through a real LiveKit call
  in this environment (no browser/mic access here). See the parent
  README's own "Current status" sections for the standard this project
  holds itself to before calling something production-validated.
- The tool registry is hand-rolled, not MCP - a deliberate v1 choice (see
  `tool_registry.py`'s docstring) with an explicit, already-verified seam
  (`Agent(mcp_servers=...)`) for later.
- `banking`'s account/transaction data is an in-memory demo store
  (`packs/banking/_bank_data.py`), not a real core-banking integration -
  it exists to prove the platform's architecture on a second vertical, not
  to be production banking software.
