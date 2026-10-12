"""Confidence (backlog D2.14): a calibration's distribution of answerable requests' best scores, the
file a server reads while it runs, ``measure`` and its twins, the retriever's field and gate, and
``toolrank calibrate`` end to end against a fake endpoint."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from toolrank.adapters.embeddings_api import OpenAIEmbeddings
from toolrank.calibration import FILE, Calibration, Calibrations, band_score, load, measure, save
from toolrank.cli import build_parser, main
from toolrank.cut import AdaptiveK
from toolrank.datasets.jsonl import write_queries, write_tools
from toolrank.domain import Query, RankedList, Tool
from toolrank.ingest.text import tool_text
from toolrank.retriever import Retriever, innermost
from toolrank.usage import UsageLog


def _cal(best, **kw):
    return Calibration(scorer="s", heads=None, instruction="find", best=tuple(sorted(best)), **kw)


def test_confidence_is_the_share_of_answerable_requests_at_or_below():
    cal = _cal([n / 10 for n in range(1, 11)], twins=(0.05, 0.12, 0.5))  # 0.1 .. 1.0
    assert cal.confidence(0.55) == 0.5 and cal.confidence(0.1) == 0.1 and cal.confidence(0.0) == 0.0
    assert cal.confidence(2.0) == 1.0
    assert cal.refused(0.2) == 0.1  # only the lowest request is below 0.2
    assert cal.caught(0.2) == pytest.approx(2 / 3) and _cal([0.5]).caught(0.2) is None
    assert band_score(cal, 0.05) == 0.1 and band_score(cal, 0.5) == 0.5


def test_save_keeps_one_entry_per_first_stage_and_a_server_follows_the_file(tmp_path):
    path = tmp_path / FILE
    save(path, _cal([0.2, 0.4]))
    save(path, Calibration("s", "abc", "find", (0.3,)))
    save(path, _cal([0.6, 0.7, 0.8]))  # replaces the first: same first stage, heads and instruction
    entries = load(path)
    assert [(e.heads, e.best) for e in entries] == [("abc", (0.3,)), (None, (0.6, 0.7, 0.8))]
    assert list(tmp_path.iterdir()) == [path]  # written whole, nothing left behind

    seen = Calibrations(path)
    assert seen.find("s", None, "find").best == (0.6, 0.7, 0.8) and seen.find("s", None, "other") is None
    save(path, _cal([0.1]))
    os.utime(path, ns=(1, 1))  # a rewrite within the same tick still changes the stamp
    assert seen.find("s", None, "find").best == (0.1,)
    path.write_text('{"version": 9}')
    assert seen.find("s", None, "find") is None and "calibration version 9" in seen.error
    path.unlink()
    assert (
        seen.find("s", None, "find") is None
        and seen.error is None
        and Calibrations(path).find("s", None, "") is None
    )


class _Lists:
    """A first stage with fixed lists per request text."""

    name = "fixed"

    def __init__(self, lists):
        self.lists, self.asked = lists, []

    def rank(self, queries, k):
        self.asked.append((k, [q.instruction for q in queries]))
        return [
            RankedList(q.id, [t for t, _ in self.lists[q.text]][:k], [s for _, s in self.lists[q.text]][:k])
            for q in queries
        ]


def test_measure_ranks_as_a_server_does_and_hides_the_gold_sources_for_twins():
    tools = [Tool(id=f"{src}/{n}", doc={"name": n}, category=src) for src in ("mail", "cal") for n in "ab"]
    lists = {
        "send": [("gone/x", 0.99), ("mail/a", 0.9), ("mail/b", 0.7), ("cal/a", 0.4), ("cal/b", 0.1)],
        "meet": [("mail/a", 0.6), ("cal/b", 0.5), ("cal/a", 0.3), ("mail/b", 0.2)],
    }
    queries = [
        Query(id="1", text="send", qrels={"mail/a": 1}, instruction="theirs"),
        Query(id="2", text="meet", qrels={"cal/b": 1}),
        Query(id="3", text="gone", qrels={"old/x": 1}),  # its tool left the catalogue
    ]
    scorer = _Lists(lists)
    best, twins, recall, skipped = measure(scorer, tools, queries, "find", depth=2)
    assert best == [0.9, 0.6] and twins == [0.4, 0.6] and recall == 1.0 and skipped == 1
    assert scorer.asked == [(2, ["find", "find"]), (4, ["find", "find"])]  # the serving instruction
    # "gone/x" is another catalogue's row in a shared index: skipped, as a server skips it
    assert measure(scorer, tools, queries[2:], "find") == ([], [], 0.0, 1)


def _hash_rows(texts):
    return [
        np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
        .standard_normal(16)
        .astype(np.float32)
        for t in texts
    ]


class _Hash:
    name = model = "hash"

    def encode(self, texts, *, kind="document"):
        return np.asarray(_hash_rows(texts))


def _catalogue(tmp_path):
    from toolrank.adapters.dense import DenseScorer

    tools = [
        Tool(id=f"s/t{i}", doc={"name": f"t{i}", "description": f"tool {i}"}, category="s") for i in range(12)
    ]
    write_tools(tmp_path / "tools.jsonl", tools)
    return lambda: DenseScorer(_Hash(), "name_desc", "instruct_query")


def test_a_matching_calibration_gives_every_search_a_confidence_and_the_gate_a_meaning(tmp_path):
    make = _catalogue(tmp_path)
    rule = AdaptiveK(margin=10.0, max_k=4)
    plain = Retriever(tmp_path, make, rule=rule, instruction="find", calibration=tmp_path / FILE)
    res = plain.search("tool 3")
    assert res.confidence is None and len(res.hits) == 4  # no file: no confidence, nothing turned away
    name = innermost(plain.state().scorer).name
    top = res.ranked[0][1]
    save(tmp_path / FILE, Calibration(name, None, "find", (top - 0.2, top - 0.1, top + 0.1, top + 0.2)))
    assert plain.search("tool 3").confidence == 0.5

    gated = Retriever(
        tmp_path, make, rule=rule, instruction="find", calibration=tmp_path / FILE, min_confidence=0.6
    )
    turned = gated.search("tool 3")
    assert turned.hits == [] and turned.confidence == 0.5 and len(turned.ranked) == 12
    assert turned.rule == "adaptive K (margin 10, max 4, min 1); confidence at least 0.6"
    asked = gated.search("tool 3", k=2)  # a request that names its own k gets what it asked for
    assert len(asked.hits) == 2 and asked.rule == "top 2" and asked.confidence == 0.5

    other = Retriever(
        tmp_path, make, rule=rule, instruction="look", calibration=tmp_path / FILE, min_confidence=0.6
    )
    res = other.search("tool 3")  # calibrated under another serving instruction: it means nothing here
    assert res.confidence is None and len(res.hits) == 4 and "confidence" not in res.rule


@pytest.fixture
def fake_endpoint(monkeypatch):
    monkeypatch.setattr(
        OpenAIEmbeddings, "_post", lambda self, texts, **kw: (_hash_rows(texts), 5 * len(texts))
    )
    monkeypatch.delenv("TOOLRANK_HEADS", raising=False)


def test_calibrate_writes_what_search_and_serve_read(tmp_path, fake_endpoint, monkeypatch, capsys):
    monkeypatch.setenv("TOOLRANK_CACHE", str(tmp_path / "no-heads"))
    tools = [
        Tool(id=f"{src}/{n}", doc={"server": src, "name": n, "description": f"{n} on {src}"},
             documentation=tool_text(src, n, f"{n} on {src}"), category=src)
        for src in ("mail", "cal", "files") for n in ("create", "list", "delete", "update")
    ]  # fmt: skip
    data, gen = tmp_path / "tools", tmp_path / "gen"
    write_tools(data / "tools.jsonl", tools)
    queries = [Query(id=f"gen-task/{i}", text=f"request {i}", qrels={tools[i % 12].id: 1}) for i in range(60)]
    write_tools(gen / "tools.jsonl", tools)
    write_queries(gen / "queries.jsonl", queries)
    base = ["--data", str(data), "--cache-dir", str(tmp_path / "c")]

    assert main(["calibrate", "--requests", str(gen), "--dry-run", *base]) == 0
    out = capsys.readouterr().out
    assert "60 requests; first stage dense/emb/toolrank-emb-v0.2/" in out and "(base heads)" in out
    assert "--min-confidence 0.05: turns away requests below" in out and "of the twins" in out
    assert not (data / FILE).exists()

    assert main(["calibrate", "--requests", str(gen), *base]) == 0
    (entry,) = load(data / FILE)
    assert len(entry.best) == 60 and len(entry.twins) == 60 and entry.requests == "gen"
    assert entry.heads is None and entry.extra == {"arm": "base", "skipped": 0}
    capsys.readouterr()

    assert main(["search", "request 3", "--json", *base]) == 0
    found = json.loads(capsys.readouterr().out)
    sure = found["confidence"]  # one of the calibration's own requests: its best score is in there
    assert abs(sure - entry.confidence(found["tools"][0]["score"])) <= 1 / 60 and sure > 0
    assert main(["search", "request 3", "--json", "--min-confidence", "0.5", *base]) == 0
    assert (json.loads(capsys.readouterr().out)["tools"] == []) == (sure < 0.5)

    write_queries(gen / "queries.jsonl", queries[:10])
    with pytest.raises(SystemExit, match="10 requests: a calibration needs at least 50"):
        main(["calibrate", "--requests", str(gen), *base])
    with pytest.raises(SystemExit, match="--requests is a dir from toolrank data gen-queries"):
        main(["calibrate", "--requests", str(tmp_path / "nowhere"), *base])
    with pytest.raises(SystemExit, match="between 0 and 1"):
        main(["search", "x", "--min-confidence", "5", *base])


def test_calibrate_from_log_measures_the_requests_a_server_answered(
    tmp_path, fake_endpoint, monkeypatch, capsys
):
    """``--from-log``: a band from generated requests describes generated requests (the LiveMCPBench
    measurement in docs/reports/backlog-d214.md), so a calibration can be built from the traffic the
    server already answered instead — its vectors come back from the embedding cache."""
    monkeypatch.setenv("TOOLRANK_CACHE", str(tmp_path / "no-heads"))
    tools = [
        Tool(id=f"{src}/{n}", doc={"server": src, "name": n, "description": f"{n} on {src}"},
             documentation=tool_text(src, n, f"{n} on {src}"), category=src)
        for src in ("mail", "cal", "files") for n in ("create", "list", "delete", "update")
    ]  # fmt: skip
    data = tmp_path / "tools"
    write_tools(data / "tools.jsonl", tools)
    base = ["--data", str(data), "--cache-dir", str(tmp_path / "c"), "--emb-url", "http://unused/v1"]

    flags = build_parser().parse_args(["calibrate", "--from-log", *base])
    enc = OpenAIEmbeddings(
        flags.emb_model, flags.emb_url, cache_dir=tmp_path / "c", truncate_prompt_tokens=flags.truncate
    )
    texts = [f"please {i} something for me" for i in range(60)]
    enc.encode(texts, kind="query")  # what serve would have cached while answering them
    log = UsageLog(data / "usage")
    events = []
    for i, text in enumerate(texts):
        called = tools[i % len(tools)].id
        events.append(
            {
                "v": 3,
                "event": "search",
                "id": f"s{i}",
                "ts": "2026-10-01T10:00:00",
                "model": flags.emb_model,
                "emb_hmac": log.digest(enc.cache_key(text)),
                "results": [[called, 0.7], [tools[(i + 1) % len(tools)].id, 0.5]],
                "shown": 2,
            }  # fmt: skip
        )
        events.append({"v": 3, "event": "call", "tool": called, "search_id": f"s{i}", "outcome": "ok"})
    (data / "usage" / "usage-2026-10-01.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )

    assert main(["calibrate", "--from-log", *base]) == 0
    out = capsys.readouterr().out
    assert "usage log: 60 requests with a linked call" in out
    assert "Recall@5" not in out  # a log has no gold list, only the tool the agent called
    (entry,) = load(data / FILE)
    assert len(entry.best) == 60 and entry.requests == "usage log, 60 requests"
    assert entry.recall_at_5 is None and len(entry.twins) == 60  # the called tool's source hidden
    assert all(0 < s <= 1 for s in entry.best) and max(entry.twins) <= max(entry.best)

    with pytest.raises(SystemExit, match="--requests DIR .generated requests. or --from-log"):
        main(["calibrate", *base])
    with pytest.raises(SystemExit, match="no usage log here"):
        main(
            [
                "calibrate",
                "--from-log",
                "--data",
                str(tmp_path / "elsewhere"),
                "--cache-dir",
                str(tmp_path / "c"),
            ]
        )
