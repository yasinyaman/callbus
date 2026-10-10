"""``scripts/memory_sim.py``: learn_sim's split, the simulated agent's memory rows, the scoring rules and a whole
run on the synthetic set with a toy encoder."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from toolrank.adapters.dense import DenseScorer
from toolrank.cli import main as cli_main
from toolrank.datasets.jsonl import load_queries, write_queries, write_tools
from toolrank.datasets.synthetic import make_queries, make_tools

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bm25_runner import _ToyEncoder  # noqa: E402


def _script(name="memory_sim"):
    path = ROOT / "scripts" / f"{name}.py"
    if not path.exists():  # the sdist ships no scripts
        pytest.skip("scripts/ is not here")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_split_is_learn_sims(tmp_path):
    mem, sim = _script(), _script("learn_sim")
    tools = make_tools(30, random.Random(0))
    queries = make_queries(tools, 25, random.Random(1))
    write_tools(tmp_path / "bench" / "tools.jsonl", tools)
    write_queries(tmp_path / "bench" / "queries.jsonl", queries)
    sim.split(tmp_path / "bench", tmp_path / "sim", 0.7, 3)
    traffic, heldout = mem.split(queries, 0.7, 3)
    assert [q.id for q in traffic] == [q.id for q in load_queries(tmp_path / "sim" / "traffic.jsonl")]
    assert [q.id for q in heldout] == [
        q.id for q in load_queries(tmp_path / "sim" / "heldout" / "queries.jsonl")
    ]


def test_the_agent_remembers_the_gold_tools_it_was_shown():
    mem = _script()
    shown, golds = [[4, 1, 2], [3, 5], [7]], [{1, 9}, {6}, {7}]
    assert mem.remember(shown, golds) == [(0, 1), (2, 7)]  # gold 9 and 6 were never shown: nothing learned
    assert mem.remember(shown, golds, noise=1.0) == [(0, 4), (1, 3), (2, 7)]  # a wrong call when one is shown


def test_memory_scores_take_each_tools_best_row_and_leave_one_out():
    mem = _script()
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    m = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    row_tool, owner = np.array([5, 2, 5]), np.array([0, 1, 1])
    tools, best = mem.memory_scores(q, m, row_tool, owner)
    assert tools.tolist() == [2, 5]
    assert np.allclose(best, [[0.6, 1.0], [0.8, 1.0]])
    _, loo = mem.memory_scores(q, m, row_tool, owner, own=np.array([0, 1]))
    assert np.allclose(loo[0], [0.6, 0.0]) and loo[1, 0] == -np.inf  # query 1 owns both rows of tool 2


def test_the_rules():
    mem = _script()
    c, m = np.array([0.5, 0.5, 0.5]), np.array([-np.inf, 0.9, 0.55])
    assert np.allclose(mem.combine(c, m, "max", 0.1, 0.0), [0.5, 0.8, 0.5])
    assert np.allclose(mem.combine(c, m, "add", 0.1, 0.5), [0.5, 0.65, 0.5])  # half way from 0.5 to 0.8
    assert np.allclose(mem.combine(c, m, "add", 0.1, 1.0), mem.combine(c, m, "max", 0.1, 0.0))
    with pytest.raises(ValueError):
        mem.combine(c, m, "mean", 0.1, 0.0)
    # a gate: rows below the floor count as none; a floor per query comes as a column
    assert np.allclose(mem.combine(c, m, "max", 0.1, 0.0, floor=0.6), [0.5, 0.8, 0.5])
    assert np.allclose(mem.combine(c, m, "max", 0.0, 0.0, floor=0.95), c)
    cc, mm = np.full((2, 2), 0.5), np.array([[0.9, 0.7], [0.9, 0.7]])
    floor = np.array([[0.8], [0.6]])
    assert np.allclose(mem.combine(cc, mm, "max", 0.0, 0.0, floor=floor), [[0.9, 0.5], [0.9, 0.7]])


def test_memory_lifts_a_tool_and_never_repeats_one():
    mem = _script()
    rng = np.random.default_rng(0)
    tv = rng.normal(size=(30, 8)).astype(np.float32)
    tv /= np.linalg.norm(tv, axis=1, keepdims=True)
    qv = tv[:4] * 0.1 + rng.normal(size=(4, 8)).astype(np.float32)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    # tool 29 remembers a row equal to query 0: under "max" it must come first for that query
    tools, best = mem.memory_scores(qv, qv[:1], np.array([29]), np.array([0]))
    r = mem.Ranker(qv, tv, tools, best)
    base_idx, _ = r.rank(None)
    idx, val = r.rank({"rule": "add", "beta": 0.0, "lambda": 1.0, "gate": None})
    assert base_idx.shape == idx.shape == (4, 30)  # 30 tools: the depth is the catalogue's
    assert idx[0, 0] == 29 and val[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert all(len(set(row.tolist())) == len(row) for row in idx)
    # gated on the first stage's best cosine plus 0.5: no row is that much better, nothing moves
    gated, _ = r.rank({"rule": "add", "beta": 0.0, "lambda": 1.0, "gate": "rel", "tau": 0.5})
    assert np.array_equal(gated, base_idx)
    # gated on the first stage's shortlist: tool 29 is not query 0's best tool, so the top 1 cannot take it in
    assert base_idx[0, 0] != 29
    short, _ = r.rank({"rule": "add", "beta": 0.0, "lambda": 1.0, "gate": "top", "tau": 1})
    assert np.array_equal(short, base_idx)
    deep, _ = r.rank({"rule": "add", "beta": 0.0, "lambda": 1.0, "gate": "top", "tau": 30})
    assert deep[0, 0] == 29


def test_a_whole_run_on_the_synthetic_set(tmp_path, monkeypatch, capsys):
    mem = _script()
    rng = random.Random(0)
    tools = make_tools(40, rng)
    queries = make_queries(tools, 120, rng)  # tools come back: the held-out queries ask some of them again
    data = tmp_path / "syn"
    write_tools(data / "tools.jsonl", tools)
    write_queries(data / "queries.jsonl", queries)
    monkeypatch.setattr(mem, "build_scorer", lambda e: DenseScorer(_ToyEncoder(), "name_desc", "plain"))
    out = tmp_path / "res" / "mem_syn"
    flags = [
        "--data",
        str(data),
        "--scorer",
        "dense",
        "--ks",
        "1,5,10",
        "--cut-margin",
        "0.2",
        "--cache-dir",
        "",
    ]
    monkeypatch.setattr(sys, "argv", ["memory_sim.py", "--out", str(out), "--", *flags])
    mem.main()
    printed = capsys.readouterr().out
    assert "**(best)**" in printed and "**(safe)**" in printed
    assert "| asked |" in printed and "| new |" in printed
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["traffic"] == 84 and report["heldout"] == 36 and report["rows"] > 0
    assert all(report[k] is None or report[k]["rule"] == "add" for k in ("picked", "picked_safe"))
    assert len(report["grid"]) == 1 + len(mem.LAMBDAS) * len(mem.BETAS) * len(mem.GATES)
    tables = report["tables"]
    assert tables["all"]["base"]["n"] == tables["asked"]["base"]["n"] + tables["new"]["base"]["n"]
    assert Path(f"{out}_memory_safe_runs.jsonl").exists()
    capsys.readouterr()
    cli_main(["compare", "--paired", f"{out}_base_runs.jsonl", f"{out}_memory_runs.jsonl"])
    assert "n = 36 queries paired by id" in capsys.readouterr().out
