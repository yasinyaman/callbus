"""Can a second stage tell that the catalogue has no tool for a request? (backlog D2.1)

Faz 2 week 5 measured the first stage's answer: a threshold on the best cosine separates answerable
requests from the same requests with their gold tools' servers removed at AUROC 0.88 (MCP-Zero) and
0.76 (LiveMCPBench), and the threshold does not carry from one set to the other. A second stage
reads the request together with each candidate, so its highest score may separate them better.

This script builds two shortlists per request — the first stage's top K over the whole catalogue
(the request has an answer) and over the catalogue without its gold tools' servers (it has none) —
scores both with a cross-encoder, and prints the AUROC and the threshold table for the
cross-encoder's highest score next to the same for the best cosine. The second shortlist has to be
built and asked again: a listwise "none" cannot be masked out of the first one.

The rule (``docs/plan/backlog.md``, D2.1): the gate becomes a default only if AUROC >= 0.90 on both
sets, and the threshold picked on MCP-Zero turns away at most 5% of LiveMCPBench's answerable
requests while catching at least 50% of the unanswerable ones.

uv run python scripts/gate_second_stage.py --rerank-url http://127.0.0.1:8095/v1 \
    --rerank-model qwen3-reranker --depth 20 --max-chars 3000 --out results/gate_livemcp.json -- \
    --data data/livemcpbench_server --scorer dense --emb-url http://127.0.0.1:8097/v1 \
    --emb-model qwen3-emb-lora --truncate 8192 --tool-format documentation \
    --query-format instruct_query --with-inst
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from routing_sweep import auroc, gate_table

from callbus.adapters.cross_encoder import CrossEncoderScorer, ScoreClient
from callbus.build import build_scorer
from callbus.cli import build_parser
from callbus.datasets.jsonl import load_queries, load_tools


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def table(name: str, have: np.ndarray, gone: np.ndarray, thresholds: list[float]) -> list[dict]:
    print(
        f"\n{name}: answerable mean {have.mean():.3f}, unanswerable (gold servers removed) mean "
        f"{gone.mean():.3f}, AUROC {auroc(have, gone):.3f}\n"
    )
    print("| threshold | answerable turned away | unanswerable caught |\n| ---: | ---: | ---: |")
    rows = []
    for th in thresholds:
        away, caught = float((have < th).mean()), float((gone < th).mean())
        rows.append({"threshold": th, "turned_away": away, "caught": caught})
        print(f"| {th:g} | {100 * away:.2f} | {100 * caught:.2f} |")
    return rows


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: gate_second_stage.py [flags] -- <callbus eval flags>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-url", required=True, help="the cross-encoder's /score endpoint")
    ap.add_argument("--rerank-model", default="qwen3-reranker")
    ap.add_argument("--rerank-template", default="qwen3", help="qwen3 or bge")
    ap.add_argument("--depth", type=int, default=20, help="shortlist the second stage reads")
    ap.add_argument("--max-chars", type=int, default=3000, help="tool documentation cut to this")
    ap.add_argument("--rerank-workers", type=int, default=4, help="calls in flight")
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries")
    ap.add_argument("--cross-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--cosine-thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv[:cut])
    e = build_parser().parse_args(["eval", *argv[cut + 1 :]])
    if e.cache_dir == "":
        e.cache_dir = None

    data = Path(e.data)
    tools = load_tools(data / "tools.jsonl")
    queries = load_queries(data / "queries.jsonl")
    if not e.with_inst:
        queries = [replace(q, instruction="") for q in queries]
    queries = [q for q in queries if q.qrels]
    if a.limit:
        queries = queries[: a.limit]

    first = build_scorer(e)
    t = first._vectors([first.tool_format(x) for x in tools], "document", first.project_tools)
    q = first._vectors([first.query_format(x) for x in queries], "query", first.project_queries)
    s = np.asarray(q @ t.T)

    names = sorted({x.category for x in tools})
    of = np.array([names.index(x.category) for x in tools])
    gold = [{names.index(x.category) for x in tools if qq.qrels.get(x.id, 0) > 0} for qq in queries]

    client = ScoreClient(
        a.rerank_model,
        a.rerank_url,
        cache_dir=e.cache_dir or ".cache/callbus",
        slots=a.rerank_workers,
    )
    cross = CrossEncoderScorer(
        client, e.tool_format or "documentation", template=a.rerank_template, max_chars=a.max_chars
    )
    print(
        f"{first.name} + {cross.name} on {data.name}: {len(queries)} requests, {len(tools)} tools, "
        f"{len(names)} servers, shortlist {a.depth}"
    )

    # the shortlists are numpy, the scoring is the endpoint: build all of them, then ask in parallel
    # (one request at a time leaves the second stage idle between its two calls)
    shortlists = []
    have_cos, gone_cos = [], []
    for i in range(len(queries)):
        row = s[i]
        masked = np.where(np.isin(of, list(gold[i])), -np.inf, row)
        top_have = np.argsort(-row)[: a.depth]
        top_gone = np.argsort(-masked)[: a.depth]
        have_cos.append(float(row[top_have[0]]))
        gone_cos.append(float(masked[top_gone[0]]))
        shortlists.append((top_have, top_gone))

    done = itertools.count(1)

    def ask(i: int) -> tuple[float, float]:
        top_have, top_gone = shortlists[i]
        out = (
            max(cross.score_tools(queries[i], [tools[j] for j in top_have])),
            max(cross.score_tools(queries[i], [tools[j] for j in top_gone])),
        )
        if (n := next(done)) % 100 == 0:
            print(f"  {n}/{len(queries)} requests scored", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=a.rerank_workers) as pool:
        pairs = list(pool.map(ask, range(len(queries))))
    have_x = [p[0] for p in pairs]
    gone_x = [p[1] for p in pairs]

    have_cos, gone_cos = np.array(have_cos), np.array(gone_cos)
    have_x, gone_x = np.array(have_x), np.array(gone_x)
    gate_table(s, of, gold, _floats(a.cosine_thresholds))  # the first stage, for reference
    cos_rows = table("best cosine, same shortlists", have_cos, gone_cos, _floats(a.cosine_thresholds))
    x_rows = table(f"{cross.name}'s highest score", have_x, gone_x, _floats(a.cross_thresholds))
    scored = client.scored()
    print(f"\nsecond stage: {scored}")

    if a.out:
        Path(a.out).write_text(
            json.dumps(
                {
                    "data": str(data),
                    "first_stage": first.name,
                    "second_stage": cross.name,
                    "requests": len(queries),
                    "tools": len(tools),
                    "depth": a.depth,
                    "max_chars": a.max_chars,
                    "auroc": {
                        "cosine": auroc(have_cos, gone_cos),
                        "cross": auroc(have_x, gone_x),
                    },
                    "means": {
                        "cosine": [float(have_cos.mean()), float(gone_cos.mean())],
                        "cross": [float(have_x.mean()), float(gone_x.mean())],
                    },
                    "scores": {
                        "cosine_have": have_cos.tolist(),
                        "cosine_gone": gone_cos.tolist(),
                        "cross_have": have_x.tolist(),
                        "cross_gone": gone_x.tolist(),
                    },
                    "thresholds": {"cosine": cos_rows, "cross": x_rows},
                    "second_stage_calls": scored,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
