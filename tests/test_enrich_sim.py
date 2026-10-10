"""``scripts/enrich_sim.py``: the written requests' parsing, each variant's text, and a whole run on the synthetic set
with a scripted chat model and a toy encoder."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from toolrank.adapters.dense import DenseScorer
from toolrank.datasets.jsonl import write_queries, write_tools
from toolrank.datasets.synthetic import make_queries, make_tools
from toolrank.domain import Tool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bm25_runner import _ToyEncoder  # noqa: E402


def _script():
    path = ROOT / "scripts" / "enrich_sim.py"
    if not path.exists():  # the sdist ships no scripts
        pytest.skip("scripts/ is not here")
    spec = importlib.util.spec_from_file_location("enrich_sim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_written_requests_are_cleaned_and_never_name_the_tool():
    m = _script()
    tool = Tool(id="s/refund_payment", doc={"name": "refund_payment", "description": "Refund a charge."})
    response = (
        '1. "Give the customer back the 40 euros we charged twice."\n'
        "2) refund ch_123 now\n"
        "- Please run refund_payment for order 77\n"  # names the tool: dropped
        "\n(4) I was billed twice for one order.\n"
        "5. I was billed twice for one order.\n"  # a repeat
        "6. A sixth one.\n7. And a seventh."
    )
    assert m.parse_requests(response, tool) == [
        "Give the customer back the 40 euros we charged twice.",
        "refund ch_123 now",
        "I was billed twice for one order.",
        "A sixth one.",
        "And a seventh.",
    ]


def test_each_variant_appends_to_the_tools_own_text():
    m = _script()
    tool = Tool(id="s/t", doc={"name": "t", "description": "d"})
    answers = {"describe/s/t": "It refunds payments.", "requests/s/t": "1. refund it\n2. give the money back"}
    v = m.texts(tool, "BASE", answers)
    assert v["describe"] == "BASE\n\nIt refunds payments."
    assert v["requests_concat"] == "BASE\n\nRequests it serves:\nrefund it\ngive the money back"
    assert v["rows"] == ["refund it", "give the money back"]
    assert m.texts(tool, "BASE", {}) == {
        "base": "BASE",
        "describe": "BASE",
        "requests_concat": "BASE",
        "rows": [],
    }
    p = m.prompts([tool])
    assert set(p) == {"describe/s/t", "requests/s/t"} and all("The tool:\n" in x for x in p.values())


class _Chat:
    """Writes the tool's own description back as its text and two requests made of its words."""

    name = "chat/scripted"

    def complete(self, system, user):
        tool = user.split("The tool:\n", 1)[1]
        words = [w for w in tool.replace(":", " ").split() if w.isalpha()][:6]
        if user.startswith("Write five"):
            return f"1. {' '.join(words[:3])}\n2. {' '.join(words[3:])}"
        return " ".join(words)


def test_a_whole_run_on_the_synthetic_set(tmp_path, monkeypatch, capsys):
    m = _script()
    rng = random.Random(0)
    tools = make_tools(40, rng)
    queries = make_queries(tools, 60, rng)
    data = tmp_path / "syn"
    write_tools(data / "tools.jsonl", tools)
    write_queries(data / "queries.jsonl", queries)
    monkeypatch.setattr(m, "build_scorer", lambda e: DenseScorer(_ToyEncoder(), "name_desc", "plain"))
    monkeypatch.setattr(m, "make_chat", lambda a: _Chat())
    out, cache = tmp_path / "res" / "enr", tmp_path / "enrich" / "syn.jsonl"
    flags = ["--data", str(data), "--scorer", "dense", "--ks", "1,5,10", "--cache-dir", ""]
    argv = ["enrich_sim.py", "--cache", str(cache), "--out", str(out), "--", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    m.main()
    printed = capsys.readouterr().out
    assert "40 descriptions" in printed and "80 requests (40 tools, 2.00 a tool)" in printed
    report = json.loads(Path(f"{out}.json").read_text())
    assert set(report["variants"]) == {"base", "describe", "requests_concat", "requests_rows"}
    assert all(v["n"] == 60 for v in report["variants"].values())
    for key in report["variants"]:
        assert Path(f"{out}_{key}_runs.jsonl").exists()
    lines = cache.read_text().splitlines()
    assert len(lines) == 80  # two prompts a tool, cached: a rerun asks nothing
    monkeypatch.setattr(m, "make_chat", lambda a: type("Refuse", (_Chat,), {"complete": None})())
    m.main()
    assert len(cache.read_text().splitlines()) == 80
