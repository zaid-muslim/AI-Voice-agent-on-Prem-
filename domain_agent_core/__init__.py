"""domain_agent_core - a domain-configurable voice agent platform.

The same runtime (core/) becomes a "hospital receptionist," a "banking
agent," or any other vertical purely by pointing it at a different domain
pack (packs/<domain>/manifest.yaml) - no forking code per domain. See
README.md for the full architecture.
"""

from __future__ import annotations
