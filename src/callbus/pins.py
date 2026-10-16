"""Tool pins: refuse a call to a tool that changed since someone approved it.

A pin is the sha256 of ``[name, description, inputSchema]`` as compact JSON with sorted keys, the
formula the hub (hub.fx42.ai) uses for its locks, so a pin from the hub and one computed here agree.
A deployment the hub renders carries ``pins.json``::

    {"hub_snapshot": "...", "servers": {SOURCE: {"hub": NAME, "version": V, "tools": {TOOL: PIN}}}}

and ``callbus serve
--pins FILE`` then checks every call before it is sent (``backends.Backends``): the tool as the
catalogue holds it (what ``search_tools`` showed) must match its pin, and so must the tool as an MCP
server listed it when the proxy connected. A source the file pins accepts only its pinned tools; a
source it does not name, or names without tools (the hub had not listed it), is called as before.
A refusal is ``refused`` in the usage log, its text starting ``pin_mismatch``.

Pure: no I/O but ``Pins.load``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MISMATCH = "pin_mismatch"


def tool_pin(name: str, description: str, schema: Any) -> str:
    blob = json.dumps([name, description, schema], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def doc_pin(doc: Mapping[str, Any]) -> str:
    """The pin of a tool record with the MCP spec's field names (``Tool.doc``, a ``tools/list``
    entry): a missing description is ``""``, a missing schema ``{}``, as the hub reads them."""
    return tool_pin(
        str(doc.get("name") or ""), str(doc.get("description") or ""), doc.get("inputSchema") or {}
    )


@dataclass(frozen=True)
class Pins:
    servers: dict[str, dict[str, str]] = field(default_factory=dict)  # source -> tool name -> pin
    snapshot: str = ""

    @classmethod
    def from_dict(cls, raw: Any) -> Pins:
        if not isinstance(raw, dict) or not isinstance(raw.get("servers"), dict):
            raise ValueError('pins: expected {"servers": {SOURCE: {"tools": {TOOL: PIN}}}}')
        servers: dict[str, dict[str, str]] = {}
        for source, entry in raw["servers"].items():
            tools = entry.get("tools") if isinstance(entry, dict) else None
            if not isinstance(tools, dict) or not all(
                isinstance(k, str) and isinstance(v, str) and len(v) == 64 for k, v in tools.items()
            ):
                raise ValueError(f"pins: {source}: expected tools as {{name: sha256 hex}}")
            servers[str(source)] = dict(tools)
        return cls(servers, str(raw.get("hub_snapshot") or ""))

    @classmethod
    def load(cls, path: str | Path) -> Pins:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"pins: cannot read {path}: {e}") from e
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hub_snapshot": self.snapshot,
            "servers": {s: {"tools": dict(t)} for s, t in sorted(self.servers.items())},
        }

    @property
    def tools(self) -> int:
        return sum(len(t) for t in self.servers.values())

    def wanted(self, source: str, name: str) -> str | None:
        """The pin a call to ``source``'s ``name`` must match; None when the source is not pinned."""
        tools = self.servers.get(source)
        return tools.get(name, "") if tools else None

    def check(self, source: str, name: str, doc: Mapping[str, Any]) -> str | None:
        """None when a call to this tool (``doc``: as the catalogue holds it) may go ahead, else why not."""
        want = self.wanted(source, name)
        if want is None:
            return None
        if not want:
            return (
                f"{MISMATCH}: {source}/{name} is not among the pinned tools of {source}; the call was refused"
            )
        if doc_pin(doc) != want:
            return (
                f"{MISMATCH}: {source}/{name} in this catalogue is not the tool that was pinned (its "
                "description or input schema changed); the call was refused"
            )
        return None

    def check_live(self, source: str, name: str, live: Mapping[str, str] | None) -> str | None:
        """The same for the tool as the MCP server listed it (``live``: name -> pin at connect)."""
        want = self.wanted(source, name)
        if not want or live is None:
            return None
        got = live.get(name)
        if got is None:
            return f"{MISMATCH}: {source} no longer lists {name}; the call was refused"
        if got != want:
            return (
                f"{MISMATCH}: {source} now describes {name} differently from the pinned tool (its "
                "description or input schema changed); the call was refused"
            )
        return None
