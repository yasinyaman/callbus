import json

import pytest

from callbus import guidance
from callbus.domain import Tool
from callbus.pins import doc_pin

ADD = {"name": "add", "description": "Add two integers.", "inputSchema": {"type": "object"}}


def _tool(doc=ADD, source="fx"):
    return Tool(id=f"{source}/{doc['name']}", doc={"server": source, **doc}, category=source)


def _note(tool="", source="fx", **kw):
    return {
        "source": source,
        "tool": tool,
        "cue": kw.get("cue", "when adding"),
        "lesson": kw.get("lesson", "pass integers, not strings"),
        "evidence": "run:42",
        "confidence": kw.get("confidence", "high"),
        "created_at": "2026-10-22T10:00:00+00:00",
    }


def _lines(*notes):
    return "".join(json.dumps(n) + "\n" for n in notes)


def test_parse_keeps_the_known_keys_and_names_a_bad_line():
    (got,) = guidance.parse(_lines({**_note(), "extra": "from a newer hub"}) + "\n")
    assert got == _note()
    for bad, why in [
        ({**_note(), "confidence": "sure"}, "confidence must be one of low, medium, high"),
        ({**_note(), "lesson": " "}, "no lesson"),
        ({**_note(), "cue": "two\nlines"}, "cue is not one line"),
        ({**_note(), "cue": "x" * 301}, "cue is not one line"),
        ({**_note(), "source": ""}, "no source"),
        ({**_note(), "tool": None}, "tool is not a string"),
        ([1], "not an object"),
    ]:
        with pytest.raises(ValueError, match=f"line 2: {why}"):
            guidance.parse(_lines(_note(), bad))
    with pytest.raises(ValueError, match="line 1: not JSON"):
        guidance.parse("{\n")


def test_a_tools_note_takes_the_pin_of_the_tool_it_came_with():
    notes = [_note(), _note("add"), _note("gone"), {**_note(), "pin": "stale"}]
    got = guidance.pinned(notes, [_tool()])
    assert got == [_note(), {**_note("add"), "pin": doc_pin(ADD)}, _note()]  # a source's note has no pin


def test_merge_replaces_the_listed_sources_and_write_drops_an_empty_file(tmp_path):
    kept = [_note(), _note(source="pets")]
    new = [_note(cue="when adding big numbers"), _note(source="other")]
    assert guidance.merge(kept, new, {"fx"}) == [_note(source="pets"), new[0]]
    assert guidance.merge(kept, [], {"fx", "pets"}) == []
    path = tmp_path / guidance.FILE
    guidance.write(path, kept)
    assert guidance.load(path) == kept and not list(tmp_path.glob("*.tmp"))
    guidance.write(path, [])
    assert not path.exists() and guidance.load(path) == []


def test_a_hit_gets_its_tools_notes_while_the_pin_holds_then_its_servers(tmp_path):
    path = tmp_path / guidance.FILE
    g = guidance.Guidance(path)
    assert g.notes(_tool()) == ()  # no file: no notes
    pin = doc_pin(ADD)
    guidance.write(
        path,
        [
            _note(cue="for the whole server", confidence="low"),
            {**_note("add"), "pin": pin},
            _note(source="pets"),
        ],
    )
    shown = {"cue": "when adding", "lesson": "pass integers, not strings", "confidence": "high"}
    server = {"cue": "for the whole server", "lesson": "pass integers, not strings", "confidence": "low"}
    assert g.notes(_tool()) == (shown, server)  # the tool's own first; no evidence, no dates
    changed = _tool({**ADD, "description": "Add two numbers."})
    assert g.notes(changed) == (server,)  # relisted with another description: its note no longer holds
    other = _tool({**ADD, "name": "sub"})
    assert g.notes(other) == (server,)

    many = [{**_note("add", cue=f"case {i}"), "pin": pin} for i in range(7)] + [
        _note(cue=f"server {i}") for i in range(7)
    ]
    guidance.write(path, many)
    got = g.notes(_tool())
    assert [n["cue"] for n in got] == [f"case {i}" for i in range(5)] + [f"server {i}" for i in range(5)]

    path.write_text("{not json\n")
    assert g.notes(_tool()) == () and "not JSON" in g.error
