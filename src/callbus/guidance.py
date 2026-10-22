"""Notes a hub's operators wrote on how to use a source or one of its tools (``DATA/guidance.jsonl``).

``callbus ingest hub`` keeps the notes the hub sent with a toolset's listing, one JSON object a
line: ``source``, ``tool`` (``""`` for a note on the whole source), ``cue`` (when it applies),
``lesson`` (what to do), ``evidence`` (``run:<id>`` or an https URL), ``confidence`` (``low``,
``medium``, ``high``) and ``created_at``, as the hub writes them, plus ``pin``: the pin of the tool
the hub listed (``pins.doc_pin``). A tool's note holds only while the tool keeps that pin, the
hub's own rule, so a tool relisted with another description or schema (``callbus ingest mcp``)
loses its notes until the hub has them again. A source's notes are replaced whenever the hub
lists the source; other sources' stay.

A search hands its tools' notes to the agent with each hit (``Guidance.notes``: the tool's own,
then its source's, ``SHOWN`` of each, ``cue``, ``lesson`` and ``confidence`` only); nothing of
them is ever in an always-on prompt or tool definition.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from callbus.domain import Tool
from callbus.pins import doc_pin

FILE = "guidance.jsonl"
CONFIDENCE = ("low", "medium", "high")
TEXT_CHARS = 300  # a cue or a lesson: one line, as the hub's rules have it
SHOWN = 5  # notes a hit carries of its tool, and of its source
_KEYS = ("source", "tool", "cue", "lesson", "evidence", "confidence", "created_at")
_SHOWN_KEYS = ("cue", "lesson", "confidence")


def _problem(note: Any) -> str | None:
    if not isinstance(note, dict):
        return "not an object"
    for key in _KEYS:
        if not isinstance(note.get(key, ""), str):
            return f"{key} is not a string"
    if not note.get("source"):
        return "no source"
    for key in ("cue", "lesson"):
        text = note.get(key, "")
        if not text.strip():
            return f"no {key}"
        if len(text) > TEXT_CHARS or "\n" in text or "\r" in text:
            return f"{key} is not one line of at most {TEXT_CHARS} characters"
    if note.get("confidence") not in CONFIDENCE:
        return f"confidence must be one of {', '.join(CONFIDENCE)}"
    return None


def parse(text: str) -> list[dict[str, str]]:
    """The notes of a ``guidance.jsonl`` (the hub's or ours), each with the known keys only;
    ``ValueError`` names the first line that is not a note."""
    out: list[dict[str, str]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            note = json.loads(line)
        except ValueError as e:
            raise ValueError(f"guidance.jsonl line {n}: not JSON ({e})") from e
        problem = _problem(note)
        if problem:
            raise ValueError(f"guidance.jsonl line {n}: {problem}")
        keep = {k: note.get(k, "") for k in _KEYS}
        if isinstance(note.get("pin"), str):
            keep["pin"] = note["pin"]
        out.append(keep)
    return out


def pinned(notes: Iterable[dict[str, str]], tools: Iterable[Tool]) -> list[dict[str, str]]:
    """``notes`` with each tool's note given the pin of the tool it names among ``tools`` (the
    listing it came with); a tool's note for a tool not among them is left out."""
    pins = {(t.category, str(t.doc.get("name") or "")): doc_pin(t.doc) for t in tools}
    out = []
    for note in notes:
        if not note["tool"]:
            out.append({k: v for k, v in note.items() if k != "pin"})
        elif (pin := pins.get((note["source"], note["tool"]))) is not None:
            out.append({**note, "pin": pin})
    return out


def load(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    return parse(path.read_text(encoding="utf-8")) if path.exists() else []


def merge(
    kept: Sequence[dict[str, str]], new: Sequence[dict[str, str]], sources: Iterable[str]
) -> list[dict[str, str]]:
    """``kept`` with the notes of ``sources`` replaced by ``new``'s (none: they have none now)."""
    replaced = set(sources)
    return [n for n in kept if n["source"] not in replaced] + [n for n in new if n["source"] in replaced]


def write(path: str | Path, notes: Sequence[dict[str, str]]) -> None:
    """Through a temp file and ``os.replace``; no notes, no file."""
    path = Path(path)
    if not notes:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(n, sort_keys=True, ensure_ascii=False) + "\n" for n in notes), encoding="utf-8"
    )
    os.replace(tmp, path)


class Guidance:
    """``DATA/guidance.jsonl`` while a server runs: read again when its size or modification time
    changes; a file that cannot be read counts as none (``error`` says why)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.error: str | None = None
        self._stamp: tuple[int, int] | None = None
        self._by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
        self._lock = threading.Lock()

    def _fresh(self) -> Mapping[tuple[str, str], list[dict[str, str]]]:
        try:
            st = self.path.stat()
            stamp: tuple[int, int] | None = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None
        if stamp != self._stamp:
            with self._lock:
                if stamp != self._stamp:
                    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
                    try:
                        for note in load(self.path) if stamp else []:
                            by_key.setdefault((note["source"], note["tool"]), []).append(note)
                        self.error = None
                    except (OSError, ValueError) as e:
                        by_key, self.error = {}, f"{type(e).__name__}: {e}"
                    self._by_key, self._stamp = by_key, stamp
        return self._by_key

    def notes(self, tool: Tool) -> tuple[dict[str, str], ...]:
        """What a search shows with ``tool``: its own notes while it keeps their pin, then its
        source's."""
        by_key = self._fresh()
        if not by_key:
            return ()
        name = str(tool.doc.get("name") or "")
        own = by_key.get((tool.category, name), []) if name else []
        if own:
            pin = doc_pin(tool.doc)
            own = [n for n in own if n.get("pin") == pin]
        found = own[:SHOWN] + by_key.get((tool.category, ""), [])[:SHOWN]
        return tuple({k: n[k] for k in _SHOWN_KEYS} for n in found)
