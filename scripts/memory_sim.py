"""Usage memory, simulated: can the requests a catalogue was already asked make it rank better, without training?

A served request whose shown list held a tool the agent then called leaves a memory row: the request's vector,
pointing at that tool. A later request scores a tool by its own cosine and by the cosine of the tool's best memory row
(``combine``); no weight changes, and a tenant's memory would be its own rows. This plays it on a benchmark the way
``scripts/learn_sim.py`` plays learning: the same seeded split (``--share`` of the queries as the traffic, the rest held
out), the same simulated agent (it calls the gold tools the served list showed: adaptive K over the first stage, so
the memory only learns what the ranking already surfaced; ``--noise P``: with probability P it calls the first shown
tool that is not gold instead, and that wrong link is remembered too), and the same breakdown of the held-out queries
(all of them; ``asked``: a gold tool of theirs was called in the traffic; ``new``: none was). The rule and its
parameters are picked on the traffic alone, each request ranked without its own rows, never on the held-out queries:
``best`` is the highest NDCG@10 there, ``safe`` the highest whose traffic requests about tools no other request had
called lose at most ``SAFE_LOSS``. A gate keeps weak rows out: from an absolute cosine, or from the first stage's best
cosine for that request plus a margin (a row less similar than the best tool is no evidence).

uv run python scripts/memory_sim.py [--share 0.7 --seed 0] [--out results/memory_toolret] -- \\
    --data data/toolret --scorer dense --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora \\
    --truncate 8192 --tool-format documentation --query-format instruct_query --with-inst --cut-margin 0.2

``--out PREFIX`` writes PREFIX.json (the grid, the picks, the tables' numbers) and
PREFIX_{base,memory,memory_safe}_runs.jsonl for ``toolrank compare --paired`` on the held-out queries.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from toolrank.build import build_scorer
from toolrank.cli import build_parser
from toolrank.cut import rule_from_flags
from toolrank.datasets.jsonl import load_queries, load_tools
from toolrank.domain import Query, RankedList
from toolrank.eval.metrics import evaluate_cut, evaluate_query
from toolrank.eval.runner import _runs_row, summarize

DEPTH = 100  # the protocol's top-100
BETAS = (
    0.0,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
)  # subtracted from a row's cosine: request-to-request cosines run higher
LAMBDAS = (1.0, 0.5, 0.25)  # the share of the way to the row's score (1: max(c, m - beta))
GATES = ((None, 0.0), ("abs", 0.7), ("abs", 0.75), ("abs", 0.8), ("abs", 0.85))
GATES += (("rel", -0.05), ("rel", 0.0), ("rel", 0.05), ("rel", 0.1))
GATES += (("top", 10), ("top", 20), ("top", 50))
SAFE_LOSS = 0.002  # the "safe" pick may cost the traffic's new-tool requests this much NDCG@10 (0.2 points)


def gold(q: Query) -> set[str]:
    return {t for t, r in q.qrels.items() if r > 0}


def split(queries: list[Query], share: float, seed: int) -> tuple[list[Query], list[Query]]:
    """learn_sim's split: queries with a gold tool, shuffled by ``seed``, the first ``share`` as traffic."""
    qs = [q for q in queries if gold(q)]
    random.Random(seed).shuffle(qs)
    n = round(len(qs) * share)
    return qs[:n], qs[n:]


def top(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the ``k`` best columns best first -> (indices, scores)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


def remember(
    shown: list[list[int]], golds: list[set[int]], noise: float = 0.0, seed: int = 0
) -> list[tuple[int, int]]:
    """The simulated agent over the traffic -> memory rows (request index, tool index): every gold tool that was
    shown, or with probability ``noise`` the first shown tool that is not gold (a wrong call, remembered as well)."""
    rng = random.Random(seed)
    rows = []
    for i, (s, g) in enumerate(zip(shown, golds, strict=True)):
        wrong = [t for t in s if t not in g]
        if noise and wrong and rng.random() < noise:
            rows.append((i, wrong[0]))
            continue
        rows.extend((i, t) for t in s if t in g)
    return rows


def memory_scores(
    q: np.ndarray, m: np.ndarray, row_tool: np.ndarray, row_owner: np.ndarray, own: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """-> (the tools that have memory rows, [queries, those tools] = the cosine of each tool's best row).
    ``own``: per query, the owner id whose rows it must not see (leave-one-out over the traffic), or -1."""
    order = np.argsort(row_tool, kind="stable")
    row_tool, m, row_owner = row_tool[order], m[order], row_owner[order]
    tools, starts = np.unique(row_tool, return_index=True)
    sims = q @ m.T
    if own is not None:
        sims[own[:, None] == row_owner[None, :]] = -np.inf
    return tools, np.maximum.reduceat(sims, starts, axis=1)


def combine(
    c: np.ndarray, mem: np.ndarray, rule: str, beta: float, lam: float, floor: Any = None
) -> np.ndarray:
    """A tool's score from its own cosine ``c`` and its best memory row's ``mem`` (-inf: none): ``max`` takes the
    row less ``beta`` when that is higher, ``add`` moves ``lam`` of the way there (``lam`` 1 is ``max``). A row
    below ``floor`` (a number, or one per query as a column) counts as none."""
    if floor is not None:
        mem = np.where(mem >= floor, mem, -np.inf)
    if rule == "max":
        return np.maximum(c, mem - beta)
    if rule == "add":
        return c + lam * np.maximum(0.0, mem - beta - c)
    raise ValueError(rule)


class Ranker:
    """Rankings of one query set under any rule: the first stage's top 100 once, the memory tools' own cosines and
    best rows once; a rule only re-scores the memory tools, which a rule can only lift."""

    def __init__(
        self, qv: np.ndarray, tv: np.ndarray, mem_tools: np.ndarray, mem: np.ndarray, chunk: int = 512
    ):
        self.idx, self.val = np.zeros((0, DEPTH), dtype=np.int64), np.zeros((0, DEPTH), dtype=np.float32)
        parts_i, parts_v = [], []
        for s in range(0, len(qv), chunk):
            i, v = top(qv[s : s + chunk] @ tv.T, DEPTH)
            parts_i.append(i)
            parts_v.append(v)
        self.idx, self.val = np.concatenate(parts_i), np.concatenate(parts_v)
        self.mem_tools, self.mem = mem_tools, mem
        self.c_mem = qv @ tv[mem_tools].T
        in_mem = np.zeros(len(tv), dtype=bool)
        in_mem[mem_tools] = True
        self.base_masked = np.where(in_mem[self.idx], -np.inf, self.val)

    def rank(self, cfg: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
        """``cfg``: rule, beta, lambda and gate (None; ``abs``: a row counts from ``tau``; ``rel``: from the first
        stage's best cosine for that query plus ``tau``; ``top``: only for the first stage's ``tau`` best tools, so
        the memory reorders the shortlist and pulls nothing in from below it). None: the first stage alone."""
        if cfg is None:
            return self.idx, self.val
        gate, tau = cfg.get("gate"), cfg.get("tau", 0.0)
        mem, floor = self.mem, None
        if gate == "abs":
            floor = tau
        elif gate == "rel":
            floor = self.val[:, :1] + tau
        elif gate == "top":
            n = min(int(tau), self.val.shape[1])
            mem = np.where(self.c_mem >= self.val[:, n - 1 : n], mem, -np.inf)
        s = combine(self.c_mem, mem, cfg["rule"], cfg["beta"], cfg["lambda"], floor)
        cand_i = np.concatenate([self.idx, np.broadcast_to(self.mem_tools, s.shape)], axis=1)
        cand_v = np.concatenate([self.base_masked, s], axis=1)
        pos, val = top(
            cand_v, self.idx.shape[1]
        )  # never deeper than the first stage: no masked entry comes back
        return np.take_along_axis(cand_i, pos, axis=1), val


def evaluate(
    queries: list[Query], idx: np.ndarray, val: np.ndarray, ids: list[str], ks: tuple[int, ...], cut: Any
) -> tuple[list[dict[str, float]], list[RankedList]]:
    rows, lists = [], []
    for q, i, v in zip(queries, idx, val, strict=True):
        r = RankedList(q.id, [ids[x] for x in i], [float(x) for x in v])
        m = evaluate_query(r.tool_ids, q.qrels, ks)
        if cut is not None:
            m.update(evaluate_cut(cut.cut(r).tool_ids, q.qrels))
        rows.append(m)
        lists.append(r)
    return rows, lists


def report(queries: list[Query], rows: list[dict[str, float]]) -> dict[str, Any]:
    s = summarize(list(zip(queries, rows, strict=True)))
    return {"n": len(queries), "overall": s.overall, "category_macro": s.category_macro}


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit(
            "usage: memory_sim.py [--share 0.7 --seed 0 --noise 0 --out PREFIX] -- <toolrank eval flags>"
        )
    at = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, default=0.7, help="of the queries played as traffic (learn_sim's)")
    ap.add_argument("--seed", type=int, default=0, help="the split's seed (learn_sim's)")
    ap.add_argument(
        "--noise", type=float, default=0.0, help="the agent's chance of calling a wrong shown tool"
    )
    ap.add_argument("--out", default=None, help="PREFIX for PREFIX.json and PREFIX_{base,memory}_runs.jsonl")
    a = ap.parse_args(argv[:at])
    e = build_parser().parse_args(["eval", *argv[at + 1 :]])
    if e.cache_dir == "":
        e.cache_dir = None
    ks = tuple(int(x) for x in e.ks.split(","))
    cut = rule_from_flags(e.cut_margin, e.cut_threshold, e.cut_max, e.cut_min, default_margin=True)

    data = Path(e.data)
    tools = load_tools(data / "tools.jsonl")
    queries = load_queries(data / "queries.jsonl")
    if not e.with_inst:
        queries = [replace(q, instruction="") for q in queries]
    traffic, heldout = split(queries, a.share, a.seed)
    scorer = build_scorer(e)
    ids = [t.id for t in tools]
    at_id = {t: n for n, t in enumerate(ids)}
    tv = scorer._vectors([scorer.tool_format(t) for t in tools], "document", scorer.project_tools)
    vec = lambda qs: scorer._vectors([scorer.query_format(q) for q in qs], "query", scorer.project_queries)  # noqa: E731
    trv, hov = vec(traffic), vec(heldout)

    # the traffic, served by the first stage alone and cut as serve cuts it; the agent calls what it was shown
    t_idx, t_val = Ranker(trv, tv, np.zeros(0, dtype=np.int64), np.zeros((len(trv), 0))).rank(None)
    shown = [
        [at_id[x] for x in cut.cut(RankedList(q.id, [ids[j] for j in i], [float(v) for v in vals])).tool_ids]
        for q, i, vals in zip(traffic, t_idx, t_val, strict=True)
    ]
    golds = [{at_id[t] for t in gold(q) if t in at_id} for q in traffic]
    rows = remember(shown, golds, a.noise, a.seed)
    if not rows:
        sys.exit("the traffic left no memory row: no gold tool was ever shown")
    owner = np.array([r[0] for r in rows])
    row_tool = np.array([r[1] for r in rows])
    m = trv[owner]
    asked = {ids[t] for t in row_tool}
    print(
        f"{scorer.name} on {data.name}: {len(tools)} tools, traffic {len(traffic)} (seed {a.seed}, share {a.share:g}, "
        f"noise {a.noise:g}), held out {len(heldout)}; {len(rows)} memory rows over {len(asked)} tools\n"
    )

    # pick the rule on the traffic: each request ranked without its own rows; a traffic request is "asked" when
    # another one left a row for one of its gold tools, the way a held-out one is
    tools_t, mem_t = memory_scores(trv, m, row_tool, owner, own=np.arange(len(traffic)))
    loo = Ranker(trv, tv, tools_t, mem_t)
    owners: dict[int, set[int]] = {}
    for o_, t_ in rows:
        owners.setdefault(t_, set()).add(o_)
    t_asked = [any(owners.get(t, set()) - {i} for t in g) for i, g in enumerate(golds)]
    grid: list[dict[str, Any] | None] = [None]
    grid += [
        {"rule": "add", "beta": b, "lambda": lam, "gate": gate, "tau": tau}
        for lam in LAMBDAS
        for b in BETAS
        for gate, tau in GATES
    ]
    scored: list[dict[str, Any]] = []
    for cfg in grid:
        r_rows, _ = evaluate(traffic, *loo.rank(cfg), ids, ks, None)
        nd = np.array([r["NDCG@10"] for r in r_rows])
        flag = np.array(t_asked)
        scored.append(
            {
                "cfg": cfg,
                "all": float(nd.mean()),
                "asked": float(nd[flag].mean()) if flag.any() else 0.0,
                "new": float(nd[~flag].mean()) if (~flag).any() else 0.0,
            }
        )
    first = scored[0]  # memory off is a candidate too: a catalogue whose requests do not repeat keeps it off
    best = max(scored, key=lambda x: x["all"])
    safe = max((x for x in scored if x["new"] >= first["new"] - SAFE_LOSS), key=lambda x: x["all"])
    print(
        f"traffic: {sum(t_asked)} requests asked about a tool another one called, {len(traffic) - sum(t_asked)} not\n"
    )
    print("| rule (traffic, leave-one-out) | NDCG@10 ↑ % | asked: fark | new: fark |")
    print("| --- | ---: | ---: | ---: |")
    shown_rows = [first, *sorted(scored[1:], key=lambda x: -x["all"])[:8]]
    shown_rows += [x for x in (best, safe) if x not in shown_rows]
    for x in shown_rows:
        label = (
            label_of(x["cfg"]) + (" **(best)**" if x is best else "") + (" **(safe)**" if x is safe else "")
        )
        print(
            f"| {label} | {100 * x['all']:.2f} | {100 * (x['asked'] - first['asked']):+.2f} | "
            f"{100 * (x['new'] - first['new']):+.2f} |"
        )

    # the held-out queries, with the memory of the whole traffic
    tools_h, mem_h = memory_scores(hov, m, row_tool, owner)
    held = Ranker(hov, tv, tools_h, mem_h)
    runs = {"base": evaluate(heldout, *held.rank(None), ids, ks, cut)}
    runs["memory"] = evaluate(heldout, *held.rank(best["cfg"]), ids, ks, cut)
    runs["memory_safe"] = evaluate(heldout, *held.rank(safe["cfg"]), ids, ks, cut)
    groups = {
        "all": list(range(len(heldout))),
        "asked": [n for n, q in enumerate(heldout) if gold(q) & asked],
        "new": [n for n, q in enumerate(heldout) if not gold(q) & asked],
    }
    out: dict[str, Any] = {
        "traffic": len(traffic),
        "heldout": len(heldout),
        "rows": len(rows),
        "tools": len(asked),
        "grid": scored,
        "picked": best["cfg"],
        "picked_safe": safe["cfg"],
    }
    metrics = ["NDCG@10", f"Recall@{ks[0]}", "Comprehensiveness@10", "K@cut", "Recall@cut"]
    labels = {"base": "first stage", "memory": f"+ memory, best ({label_of(best['cfg'])})"}
    labels["memory_safe"] = f"+ memory, safe ({label_of(safe['cfg'])})"
    print(f"\n| held out | n | ranking | {' | '.join(metrics)} | NDCG@10 cat-macro |")
    print("| --- | ---: | --- | " + " | ".join("---:" for _ in metrics) + " | ---: |")
    for g, members in groups.items():
        qs = [heldout[n] for n in members]
        for key, (rws, _) in runs.items():
            rep = (
                report(qs, [rws[n] for n in members]) if qs else {"n": 0, "overall": {}, "category_macro": {}}
            )
            out.setdefault("tables", {}).setdefault(g, {})[key] = rep
            o, cm = rep["overall"], rep["category_macro"]
            cells = [
                (f"{o[k]:.2f}" if k == "K@cut" else f"{100 * o[k]:.2f}") if k in o else "—" for k in metrics
            ]
            macro = f"{100 * cm['NDCG@10']:.2f}" if cm else "—"
            print(f"| {g} | {rep['n']} | {labels[key]} | {' | '.join(cells)} | {macro} |")
    if a.out:
        prefix = Path(a.out)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".json").write_text(json.dumps(out, indent=2))
        for key, (rws, lists) in runs.items():
            rws_ = [_runs_row(q, r, mm) for q, r, mm in zip(heldout, lists, rws, strict=True)]
            cfg = {"memory": best["cfg"], "memory_safe": safe["cfg"]}.get(key)
            name = scorer.name + ("" if cfg is None else f"+memory[{label_of(cfg)}]")
            Path(f"{a.out}_{key}_runs.jsonl").write_text(
                json.dumps({"dataset": f"{data.name}_heldout", "scorer": name, "rows": rws_}, indent=2)
            )
        print(f"\nsaved {prefix.with_suffix('.json')} and {a.out}_{{base,memory,memory_safe}}_runs.jsonl")


def label_of(cfg: dict[str, Any] | None) -> str:
    if cfg is None:
        return "first stage alone"
    rule = "max" if cfg["lambda"] == 1.0 else f"λ {cfg['lambda']:g}"
    gate = {
        None: "",
        "abs": f", row ≥ {cfg.get('tau', 0):g}",
        "rel": f", row ≥ best tool {cfg.get('tau', 0):+g}",
        "top": f", first stage's top {cfg.get('tau', 0):g} only",
    }[cfg.get("gate")]
    return f"{rule} β {cfg['beta']:g}{gate}"


if __name__ == "__main__":
    main()
