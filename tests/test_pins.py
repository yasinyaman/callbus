import dataclasses
import json
import sys
from pathlib import Path

import pytest

from callbus.pins import MISMATCH, Pins, doc_pin, tool_pin

FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_server.py")


def test_the_pin_is_the_hubs_formula():
    """Values from the hub's own ``domain.tool_pin``: a lock and callbus must agree on every pin."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }
    assert tool_pin("add", "Add two integers.", schema) == (
        "60dfc211ee0ba57f96955963ab8b0f561e1f6526b729262963c37b6406732b7a"
    )
    assert (
        tool_pin("zaman", "Şu anki saat", {})
        == "911ee4eb0a089d3a9fe8ec38093b7a0a82c3e8b1153804a032adeb1679cdaa67"
    )
    # a record without a description or a schema reads as the hub reads it: "" and {}
    assert doc_pin({"name": "zaman", "description": "Şu anki saat"}) == tool_pin("zaman", "Şu anki saat", {})
    assert doc_pin({"name": "x", "title": "X", "inputSchema": {}}) == tool_pin("x", "", {})


def test_pins_file_shapes_and_checks(tmp_path):
    good = tool_pin("add", "Add.", {})
    raw = {
        "hub_snapshot": "2026-10-16T04:09:53+00:00",
        "servers": {"fx": {"hub": "io.x/fx", "tools": {"add": good}}, "later": {"tools": {}}},
    }
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(raw))
    pins = Pins.load(path)
    assert (pins.tools, pins.snapshot) == (1, "2026-10-16T04:09:53+00:00")
    assert pins.check("fx", "add", {"name": "add", "description": "Add."}) is None
    changed = pins.check("fx", "add", {"name": "add", "description": "Add, and send the sum elsewhere."})
    assert changed.startswith(MISMATCH) and "changed" in changed
    assert "not among the pinned tools" in pins.check("fx", "pid", {"name": "pid"})
    assert pins.check("other", "anything", {"name": "anything"}) is None  # not pinned
    assert (
        pins.check("later", "anything", {"name": "anything"}) is None
    )  # pinned without tools: the hub had none
    assert pins.check_live("fx", "add", {"add": good}) is None
    assert "no longer lists add" in pins.check_live("fx", "add", {})
    assert "describes add differently" in pins.check_live("fx", "add", {"add": "0" * 64})
    assert Pins.from_dict(pins.to_dict()).servers == pins.servers
    for bad in ({}, {"servers": []}, {"servers": {"fx": {"tools": {"add": "short"}}}}):
        with pytest.raises(ValueError, match="pins"):
            Pins.from_dict(bad)
    (tmp_path / "broken.json").write_text("{")
    with pytest.raises(ValueError, match="cannot read"):
        Pins.load(tmp_path / "broken.json")


def test_serve_refuses_calls_to_tools_that_changed_since_the_pins():
    pytest.importorskip("mcp")
    import anyio

    from callbus.adapters.backends import Backends
    from callbus.adapters.mcp_client import fetch_many
    from callbus.ingest.mcp import ServerConfig

    cfg = ServerConfig("fx", "stdio", command=sys.executable, args=(FIXTURE, "--mode", "extra"))
    listed = {t.doc["name"]: t for t in anyio.run(lambda: fetch_many([cfg], timeout=30))["fx"]}
    add, pid = listed["add"], listed["pid"]

    def pins_of(*tools):
        return Pins({"fx": {t.doc["name"]: doc_pin(t.doc) for t in tools}})

    def call(pins, tool, args):
        async def main():
            async with Backends([cfg], pins=pins, connect_timeout=30).running() as b:
                return await b.call(tool, args)

        return anyio.run(main)

    ok = call(pins_of(add, pid), add, {"a": 2, "b": 3})
    assert (ok.outcome, ok.result.content[0].text) == ("ok", "5")

    # the catalogue and the pins agree, the server does not: refused before the call is sent
    told = dataclasses.replace(add, doc={**add.doc, "description": "Add two integers, faster."})
    out = call(pins_of(told), told, {"a": 1, "b": 1})
    assert out.outcome == "refused" and "now describes add differently" in out.result.content[0].text

    # the server matches the pins, the catalogue that search showed does not
    out = call(pins_of(add), told, {"a": 1, "b": 1})
    assert out.outcome == "refused" and "not the tool that was pinned" in out.result.content[0].text

    # a pinned source's tool the pins do not name
    out = call(pins_of(add), pid, {})
    assert out.outcome == "refused" and "not among the pinned tools" in out.result.content[0].text

    # without pins, or for a source the file does not pin, calls go as before
    assert call(None, pid, {}).outcome == "ok"
    assert call(Pins({"elsewhere": {"x": "0" * 64}}), pid, {}).outcome == "ok"
