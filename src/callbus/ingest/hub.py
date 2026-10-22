"""The hub (hub.fx42.ai) as a tool source: the tools it has listed for the servers of a toolset.

``POST {hub}/api/v1/deploy/ingest`` with a ``toolset.yaml`` body answers ``{"tools.jsonl": ...,
"sources.json": ...}``, callbus's own ingest format: per server of the toolset the tools the hub
has listed, and per source its kind (``mcp`` or ``openapi``), its transport or, for an API, the spec
and base URL. ``listings`` turns that into one ``sync.Listing`` per source (a server the hub has not
listed yet is a source with 0 tools, which a later ``callbus ingest mcp`` fills) and the pins of what
came back, which ``callbus serve --pins`` then holds the servers to. A newer hub may add
``guidance.jsonl``, the notes its operators wrote on the toolset's servers and tools (``notes``;
``callbus.guidance``); an older one, or a toolset with none, sends no such key.

Each tool's indexed text is recomputed with ``ingest.text.tool_text``, so a tool reads the same
whether it came from the hub or from ``callbus ingest mcp`` (and hits the same cache entries).
The hub sends no credentials, and the only one sent to it is its own: the hub's API answers a
token its operators give out (``CALLBUS_HUB_TOKEN``, sent as ``Authorization: Bearer``; over https
only, or to a hub on this machine). Calls still need the servers in ``serve --config`` (the hub's
deploy page writes a ``callbus.json`` for the same toolset).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any

from callbus import guidance
from callbus.datasets.jsonl import tools_from_lines
from callbus.domain import Tool
from callbus.ingest.sync import Listing
from callbus.ingest.text import MAX_CHARS, tool_text
from callbus.pins import Pins, doc_pin

ENDPOINT = "/api/v1/deploy/ingest"
MAX_BYTES = 64 * 1024 * 1024
KINDS = ("mcp", "openapi")
TOKEN_ENV = "CALLBUS_HUB_TOKEN"
_LOOPBACK = ("localhost", "127.0.0.1", "::1")
_MANIFEST_ONLY = ("kind", "tools", "synced_at")  # sync writes these itself


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # a POST is never replayed somewhere else; the 3xx comes back as an error


def endpoint(hub: str) -> str:
    """``https://hub.fx42.ai`` (or a URL already ending in the endpoint) -> the ingest endpoint."""
    hub = hub.strip().rstrip("/")
    if not re.match(r"https?://[^/]", hub):
        raise ValueError(f"the hub's address must be an http(s) URL, got {hub!r}")
    return hub if hub.endswith(ENDPOINT) else hub + ENDPOINT


def fetch(hub: str, toolset: bytes, *, timeout: float = 120.0, token: str = "") -> dict[str, str]:
    """The hub's ingest files for ``toolset`` (the YAML as written); ``ValueError`` names what failed.
    ``token`` is the hub's API token: sent over https only, or to a hub on this machine."""
    from callbus import __version__

    url, token = endpoint(hub), token.strip()
    headers = {
        "Content-Type": "application/yaml",
        "Accept": "application/json",
        "User-Agent": f"callbus/{__version__}",
    }
    if token:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https" and parts.hostname not in _LOOPBACK:
            raise ValueError(f"{url}: the hub's token goes over https only")
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=toolset, method="POST", headers=headers)
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout) as r:
            status, body = r.status, r.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read(64 * 1024)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise ValueError(f"{url}: the hub did not answer ({reason})") from e
    if len(body) > MAX_BYTES:
        raise ValueError(f"{url}: the answer is larger than {MAX_BYTES // 2**20} MiB")
    if status != 200:
        try:
            detail = json.loads(body).get("detail")
        except (ValueError, AttributeError):
            detail = None
        hint = ""
        if status == 401:  # the hub's API is closed without a token
            hint = (
                f" (the hub refused {TOKEN_ENV})"
                if token
                else f" (set {TOKEN_ENV} to one from its operators)"
            )
        raise ValueError(f"{url}: HTTP {status}" + (f": {detail}" if detail else "") + hint)
    try:
        files = json.loads(body)
    except ValueError as e:
        raise ValueError(f"{url}: not JSON") from e
    if not isinstance(files, dict) or not all(
        isinstance(files.get(k), str) for k in ("tools.jsonl", "sources.json")
    ):
        raise ValueError(f'{url}: expected {{"tools.jsonl": ..., "sources.json": ...}}')
    if not isinstance(files.get(guidance.FILE, ""), str):
        raise ValueError(f"{url}: {guidance.FILE} is not text")
    return files


def _retext(source: str, t: Tool) -> Tool:
    doc = {**t.doc, "server": source}
    name = str(doc.get("name") or "")
    if not name or t.id != f"{source}/{name}":
        raise ValueError(f"hub: tool {t.id!r} is not {source}/<its name>")
    description = str(doc.get("description") or doc.get("title") or "")
    text = tool_text(source, name, description, doc.get("inputSchema"), max_chars=MAX_CHARS)
    return Tool(id=t.id, doc=doc, documentation=text, category=source)


def listings(files: dict[str, str]) -> tuple[list[Listing], Pins]:
    """One listing per source the hub named, and the pins of every tool it sent."""
    try:
        manifest = json.loads(files["sources.json"])
        tools = tools_from_lines(files["tools.jsonl"].splitlines())
    except (ValueError, KeyError) as e:
        raise ValueError(f"hub: unreadable ingest files ({e})") from e
    if not isinstance(manifest, dict):
        raise ValueError("hub: sources.json is not an object")
    by_source: dict[str, list[Tool]] = defaultdict(list)
    for t in tools:
        by_source[t.category].append(t)
    strays = sorted(set(by_source) - set(manifest))
    if strays:
        raise ValueError(f"hub: tools of sources sources.json does not name: {strays[:3]}")
    out: list[Listing] = []
    pins: dict[str, dict[str, str]] = {}
    snapshot = ""
    for name, entry in sorted(manifest.items()):
        if not isinstance(entry, dict) or entry.get("kind") not in KINDS:
            raise ValueError(f"hub: source {name!r}: kind must be one of {', '.join(KINDS)}")
        own = [_retext(name, t) for t in by_source.get(name, [])]
        info = {k: v for k, v in entry.items() if k not in _MANIFEST_ONLY}
        out.append(Listing(name, entry["kind"], own, info=info))
        pins[name] = {str(t.doc["name"]): doc_pin(t.doc) for t in own}
        snapshot = snapshot or str(entry.get("synced_at") or "")
    return out, Pins(pins, snapshot)


def notes(files: dict[str, str], listings: list[Listing]) -> list[dict[str, str]]:
    """The hub's notes on the sources of ``listings``, each tool's with the pin of the tool as the
    hub listed it (a note on a tool the listing lacks is left out); none without ``guidance.jsonl``."""
    try:
        found = guidance.parse(files.get(guidance.FILE, ""))
    except ValueError as e:
        raise ValueError(f"hub: {e}") from e
    names = {listing.name for listing in listings}
    strays = sorted({n["source"] for n in found} - names)
    if strays:
        raise ValueError(f"hub: notes on sources sources.json does not name: {strays[:3]}")
    return guidance.pinned(found, (t for listing in listings for t in listing.tools or []))
