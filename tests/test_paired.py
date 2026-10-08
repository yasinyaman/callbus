import json
import random

import pytest

from toolrank.cli import main
from toolrank.datasets.jsonl import write_queries, write_tools
from toolrank.datasets.synthetic import make_queries, make_tools
from toolrank.domain import Query
from toolrank.eval.paired import category_weights, permutation_test, sign_test, weighted_permutation_test


def test_sign_test_known_example():
    # 8 wins, 2 losses: two-sided binomial p = 2 * (C(10,0) + C(10,1) + C(10,2)) / 2**10
    wins, losses, p = sign_test([1.0] * 8 + [0.0] * 2, [0.0] * 8 + [1.0] * 2)
    assert (wins, losses) == (8, 2)
    assert p == pytest.approx(2 * (1 + 10 + 45) / 1024)


def test_sign_test_drops_ties_and_checks_lengths():
    assert sign_test([1.0, 1.0, 2.0], [1.0, 0.0, 2.0]) == (1, 0, 1.0)  # two ties drop out
    with pytest.raises(ValueError):
        sign_test([1.0], [1.0, 2.0])


def test_permutation_test_is_exact_up_to_twelve_queries():
    # differences +1, +1: of the 4 sign assignments only ++ and -- reach |sum| 2 -> p = 0.5
    assert permutation_test([1.0, 1.0], [0.0, 0.0]) == 0.5
    assert permutation_test([1.0], [1.0]) == 1.0  # no difference: every assignment reaches


def test_permutation_test_is_deterministic_and_spots_a_real_difference():
    b = [0.7, 0.6, 0.5, 0.75, 0.4, 0.65, 0.55, 0.7, 0.6, 0.5, 0.68, 0.57, 0.46, 0.72]
    a = [x + 0.2 for x in b]  # 14 queries: above the exact range, Monte Carlo
    assert permutation_test(a, b, seed=1) == permutation_test(a, b, seed=1)
    assert permutation_test(a, b, seed=1) < 0.01
    with pytest.raises(ValueError):
        permutation_test([1.0], [1.0, 2.0])


def _eval_runs(tmp_path, name, *extra):
    rng = random.Random(3)
    tools = make_tools(60, rng)
    queries = make_queries(tools, 30, rng)
    data = tmp_path / name
    data.mkdir()
    write_tools(data / "tools.jsonl", tools)
    write_queries(data / "queries.jsonl", queries)
    out = tmp_path / f"{name}.runs.json"
    main(["eval", "--data", str(data), "--scorer", "bm25", *extra, "--runs-out", str(out)])
    return json.loads(out.read_text())


def test_eval_runs_out_writes_one_row_per_query(tmp_path):
    r = _eval_runs(tmp_path, "d")
    assert r["scorer"].startswith("bm25") and r["dataset"] == "d"
    assert len(r["rows"]) == 30
    row = r["rows"][0]
    assert set(row) == {
        "id",
        "task",
        "top",
        "P@1",
        "hit@5",
        "NDCG@10",
    }  # synthetic queries: a task, no category
    assert len(row["top"]) <= 20 and row["P@1"] in (0.0, 1.0)
    # P@1 and hit@5 agree with the ids the row carries
    queries = {
        q["id"]: {d for d, g in q["qrels"].items() if g > 0}
        for q in map(json.loads, (tmp_path / "d" / "queries.jsonl").read_text().splitlines())
    }
    for row in r["rows"]:
        rel = queries[row["id"]]
        assert row["P@1"] == float(row["top"][:1] != [] and row["top"][0] in rel)
        assert row["hit@5"] == float(any(t in rel for t in row["top"][:5]))


def _write_runs(path, scorer, rows):
    path.write_text(json.dumps({"dataset": "d", "scorer": scorer, "rows": rows}))


def test_compare_paired_reports_sign_and_permutation_p(tmp_path, capsys):
    ids = [f"q{i}" for i in range(10)]
    rows_a = [{"id": i, "top": [], "P@1": 1.0, "hit@5": 1.0, "NDCG@10": 0.8} for i in ids]
    rows_b = [{**r, "P@1": 0.0, "NDCG@10": 0.6} for r in rows_a[:8]] + rows_a[8:]
    _write_runs(tmp_path / "a.json", "alpha", rows_a)
    _write_runs(tmp_path / "b.json", "beta", rows_b)
    main(["compare", "--paired", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    out = capsys.readouterr().out
    # P@1: 8 up, 0 down -> p = 2/2**8; NDCG@10: the 2 ties copy each extreme assignment 4x
    assert "| P@1 (8 up, 0 down) | 100.00 | 20.00 | +80.00 | 0.0078 |" in out
    assert "| hit@5 (0 up, 0 down) | 100.00 | 100.00 | +0.00 | 1.0000 |" in out
    assert "| NDCG@10 (mean diff) | 80.00 | 64.00 | +16.00 | 0.0078 |" in out
    assert "n = 10 queries paired by id" in out


def test_compare_paired_needs_two_runs_files(tmp_path):
    _write_runs(tmp_path / "a.json", "alpha", [])
    _write_runs(tmp_path / "b.json", "beta", [])
    with pytest.raises(SystemExit):  # three files
        main(["compare", "--paired", "a", "b", "c"])
    with pytest.raises(SystemExit):  # no shared query id
        main(["compare", "--paired", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    (tmp_path / "r.json").write_text(json.dumps({"scorer": "x", "overall": {}}))  # a results file
    with pytest.raises(SystemExit):
        main(["compare", "--paired", str(tmp_path / "r.json"), str(tmp_path / "a.json")])


def test_data_seed_reaches_the_split_not_the_training_seed(monkeypatch, tmp_path):
    import toolrank.finetune as ft

    captured = {}

    def fake_run(job, log=print):
        captured["split_seed"] = job.seed
        captured["train_seed"] = job.train.seed
        return {"embed_only": True}

    monkeypatch.setattr(ft, "run", fake_run)
    args = [
        "finetune",
        "--data",
        str(tmp_path / "pairs.jsonl"),
        "--dev",
        str(tmp_path / "dev"),
        "--out",
        str(tmp_path / "h.npz"),
        "--embed-only",
        "--results",
        str(tmp_path),
    ]
    main(args + ["--seed", "5", "--data-seed", "11"])
    assert captured == {"split_seed": 11, "train_seed": 5}
    main(args + ["--seed", "5"])  # without --data-seed the split follows --seed, as before
    assert captured == {"split_seed": 5, "train_seed": 5}


def test_compare_paired_refuses_repeated_ids_and_warns_about_two_datasets(tmp_path, capsys):
    rows = [{"id": i, "top": [], "P@1": 1.0, "hit@5": 1.0, "NDCG@10": 0.8} for i in ("q1", "q2", "q1")]
    _write_runs(tmp_path / "a.json", "alpha", rows)
    _write_runs(tmp_path / "b.json", "beta", rows[:2])
    with pytest.raises(SystemExit, match="query ids repeat"):
        main(["compare", "--paired", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    other = json.loads((tmp_path / "b.json").read_text())
    (tmp_path / "c.json").write_text(json.dumps({**other, "dataset": "another_set"}))
    main(["compare", "--paired", str(tmp_path / "b.json"), str(tmp_path / "c.json")])
    assert "pairing by id assumes one set" in capsys.readouterr().err


def test_weighted_permutation_test_is_exact_small_and_drops_zero_weights():
    # weighted differences +0.125 and +0.5 (the third query weighs 0): only ++ and -- reach 0.625
    assert weighted_permutation_test([1.0, 1.0, 1.0], [0.5, 0.0, 0.0], [0.25, 0.5, 0.0]) == 0.5
    assert weighted_permutation_test([1.0, 0.0], [0.0, 1.0], [0.0, 0.0]) == 1.0  # nothing weighs
    with pytest.raises(ValueError):
        weighted_permutation_test([1.0], [1.0], [1.0, 1.0])


def test_weighted_permutation_test_is_deterministic_and_spots_a_real_difference():
    b = [0.7, 0.6, 0.5, 0.75, 0.4, 0.65, 0.55, 0.7, 0.6, 0.5, 0.68, 0.57, 0.46, 0.72]
    a = [x + 0.2 for x in b]
    w = [1.0 / len(b)] * len(b)
    assert weighted_permutation_test(a, b, w, seed=1) == weighted_permutation_test(a, b, w, seed=1)
    assert weighted_permutation_test(a, b, w, seed=1) < 0.01
    noise = [0.05 if i % 2 else -0.05 for i in range(len(b))]
    assert weighted_permutation_test([x + e for x, e in zip(b, noise, strict=True)], b, w) > 0.5


def test_category_weights_reproduce_the_category_averages():
    from toolrank.eval.metrics import aggregate, category_macro

    groups = {"q1": ("t1", "x"), "q2": ("t1", "x"), "q3": ("t2", "x"), "q4": ("t3", "y")}
    ids = list(groups)
    scores = {"q1": 0.5, "q2": 1.0, "q3": 0.0, "q4": 1.0}
    w = category_weights(groups, ids)
    per_task = {
        t: aggregate([{"NDCG@10": scores[i]} for i in ids if groups[i][0] == t]) for t in ("t1", "t2", "t3")
    }
    per_cat, macro = category_macro(per_task, {"t1": "x", "t2": "x", "t3": "y"})
    for c in ("x", "y"):
        assert sum(v * scores[i] for v, i in zip(w[c], ids, strict=True)) == pytest.approx(
            per_cat[c]["NDCG@10"]
        )
    assert sum(v * scores[i] for v, i in zip(w["cat-macro"], ids, strict=True)) == pytest.approx(
        macro["NDCG@10"]
    )
    assert all(sum(v) == pytest.approx(1.0) for v in w.values())


def _category_rows():
    groups = {"q1": ("t1", "x"), "q2": ("t1", "x"), "q3": ("t2", "x"), "q4": ("t3", "y")}
    b_scores = {"q1": 0.5, "q2": 1.0, "q3": 0.0, "q4": 1.0}
    rows_a = [{"id": i, "top": [], "P@1": 1.0, "hit@5": 1.0, "NDCG@10": 1.0} for i in groups]
    rows_b = [{**r, "NDCG@10": b_scores[r["id"]]} for r in rows_a]
    return groups, rows_a, rows_b


def test_compare_paired_adds_category_rows_from_the_runs_rows(tmp_path, capsys):
    groups, rows_a, rows_b = _category_rows()

    def tag(rows):
        return [{**r, "task": groups[r["id"]][0], "category": groups[r["id"]][1]} for r in rows]

    _write_runs(tmp_path / "a.json", "alpha", tag(rows_a))
    _write_runs(tmp_path / "b.json", "beta", tag(rows_b))
    main(["compare", "--paired", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    out = capsys.readouterr().out
    # x: tasks t1 (0.75) and t2 (0.0) -> 37.5; y: t3 -> 100; cat-macro 68.75; 4 queries: exact p
    assert "| NDCG@10 x (2 tasks) | 100.00 | 37.50 | +62.50 | 0.5000 |" in out
    assert "| NDCG@10 y (1 tasks) | 100.00 | 100.00 | +0.00 | 1.0000 |" in out
    assert "| NDCG@10 cat-macro (3 tasks) | 100.00 | 68.75 | +31.25 | 0.5000 |" in out


def test_compare_paired_takes_categories_from_a_dataset_dir(tmp_path, capsys):
    groups, rows_a, rows_b = _category_rows()
    _write_runs(tmp_path / "a.json", "alpha", rows_a)
    _write_runs(tmp_path / "b.json", "beta", rows_b)
    main(["compare", "--paired", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert "cat-macro" not in capsys.readouterr().out  # rows without categories: no category rows
    data = tmp_path / "set"
    data.mkdir()
    write_queries(
        data / "queries.jsonl",
        [Query(id=i, text=i, qrels={"t": 1}, task=t, category=c) for i, (t, c) in groups.items()],
    )
    main(["compare", "--paired", "--data", str(data), str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert "| NDCG@10 cat-macro (3 tasks) | 100.00 | 68.75 | +31.25 | 0.5000 |" in capsys.readouterr().out
    write_queries(
        data / "queries.jsonl", [Query(id="q1", text="q1", qrels={"t": 1}, task="t1", category="x")]
    )
    with pytest.raises(SystemExit, match="not in"):
        main(["compare", "--paired", "--data", str(data), str(tmp_path / "a.json"), str(tmp_path / "b.json")])
