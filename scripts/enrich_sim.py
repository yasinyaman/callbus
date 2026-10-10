"""LLM-written text for each tool, indexed without training: does the backbone rank better with it?

Two kinds of text, written once per tool by a chat model (cached in ``--cache``, keyed by model and prompt) and
scored with the backbone the ``toolrank eval`` flags after ``--`` describe, over every query of the set:

- ``describe``: a description for the search index (what the tool does, the requests and situations it serves, when
  another tool fits better; nothing the documentation does not state, no example call), appended to the tool's text;
- ``requests``: five requests the tool fulfils, in different styles, never naming it. ``requests_concat`` appends
  them to the tool's text (doc2query); ``requests_rows`` embeds each as a row of its own on the document side, and a
  tool scores by its best row, its own text being one of them (the usage memory's ``max`` with written requests).

Nothing is tuned: three rankings next to the first stage alone, with per-query rows for ``toolrank compare
--paired``. The writer must not be the model that wrote the set's queries, or a written request matches a written
query: gen-queries' sets and MCP-Zero's queries are Qwen3-8B's, ToolRet's and LiveMCPBench's are not. Appended text
sits after the documentation, so a document longer than ``--truncate`` loses it.

uv run python scripts/enrich_sim.py --cache data/enrich/livemcpbench_server.jsonl --out results/enrich_livemcpbench \\
    -- --data data/livemcpbench_server --scorer dense --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora \\
    --truncate 8192 --emb-batch 128 --tool-format documentation --query-format instruct_query --with-inst
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_sim import DEPTH, Ranker, evaluate, report, top  # noqa: E402

from toolrank.build import build_scorer  # noqa: E402
from toolrank.cli import build_parser  # noqa: E402
from toolrank.datasets.genqueries import TOOL_CHARS, _text, clean, names_tool  # noqa: E402
from toolrank.datasets.jsonl import load_queries, load_tools  # noqa: E402
from toolrank.datasets.mcp_zero import generate  # noqa: E402
from toolrank.domain import Tool  # noqa: E402
from toolrank.eval.runner import _runs_row  # noqa: E402

SYSTEM = (
    "You write the text a search index keeps for a software tool, so that the requests it serves find it. "
    "Answer with the text only: no heading, no label, no explanation."
)
DESCRIBE = (
    "Write three to five sentences of plain language about this tool for a search index: what it does, the kinds "
    "of requests and situations it serves, and when another tool would fit better. Use only what the documentation "
    "states or plainly implies: do not invent capabilities, parameters or limits. Do not write example calls."
)
REQUESTS = (
    "Write five different requests that this tool fulfils, one per line and nothing else: (1) a person asking an "
    "assistant to get something done, with plausible specifics; (2) the terse note an agent writes to itself for its "
    "next step; (3) a goal in everyday words from someone who does not know the system; (4) a situation or problem "
    "the tool resolves, without naming the operation; (5) a short question. Do not use the tool's name and do not "
    "copy phrases of its description."
)
N_REQUESTS = 5
_NUMBER = re.compile(r"^\s*(?:\(?\d+[.)]|[-*•])\s*")


def prompts(tools: list[Tool]) -> dict[str, str]:
    """The two prompts of every tool, keyed ``describe/<id>`` and ``requests/<id>``."""
    out = {}
    for t in tools:
        text = _text(t, TOOL_CHARS)
        out[f"describe/{t.id}"] = f"{DESCRIBE}\n\nThe tool:\n{text}"
        out[f"requests/{t.id}"] = f"{REQUESTS}\n\nThe tool:\n{text}"
    return out


def parse_requests(response: str, tool: Tool) -> list[str]:
    """The requests in a response: one a line, numbering and quotes removed, none that names the tool, at most five."""
    out: list[str] = []
    for line in clean(response).splitlines():
        r = clean(_NUMBER.sub("", line))
        if r and not names_tool(r, tool) and r not in out:
            out.append(r)
    return out[:N_REQUESTS]


def texts(tool: Tool, base: str, responses: dict[str, str]) -> dict[str, Any]:
    """A tool's indexed texts under each variant; a tool the model did not answer for keeps its own text."""
    desc = clean(responses.get(f"describe/{tool.id}", ""))
    reqs = parse_requests(responses.get(f"requests/{tool.id}", ""), tool)
    return {
        "base": base,
        "describe": f"{base}\n\n{desc}" if desc else base,
        "requests_concat": f"{base}\n\nRequests it serves:\n" + "\n".join(reqs) if reqs else base,
        "rows": reqs,
    }


def rows_rank(
    qv: np.ndarray, tv: np.ndarray, rv: np.ndarray, row_tool: np.ndarray, k: int = DEPTH, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Each tool scored by the best of its own text and its rows -> the top ``k`` per query. Chunked over the
    queries, so a catalogue whose every tool has rows never needs the [queries, rows] matrix whole."""
    order = np.argsort(row_tool, kind="stable")
    rv, row_tool = rv[order], row_tool[order]
    tools, starts = np.unique(row_tool, return_index=True)
    parts_i, parts_v = [], []
    for s in range(0, len(qv), chunk):
        q = qv[s : s + chunk]
        scores = q @ tv.T
        if len(rv):
            scores[:, tools] = np.maximum(scores[:, tools], np.maximum.reduceat(q @ rv.T, starts, axis=1))
        i, v = top(scores, k)
        parts_i.append(i)
        parts_v.append(v)
    return np.concatenate(parts_i), np.concatenate(parts_v)


def make_chat(a: argparse.Namespace) -> Any:
    from toolrank.adapters.chat_api import OpenAIChat

    return OpenAIChat(a.gen_model, a.gen_url, max_tokens=a.gen_max_tokens, extra_body=json.loads(a.gen_extra))


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: enrich_sim.py --cache FILE [--out PREFIX] [chat flags] -- <toolrank eval flags>")
    at = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="the chat model's answers (JSONL), reused on a rerun")
    ap.add_argument("--out", default=None, help="PREFIX for PREFIX.json and PREFIX_<variant>_runs.jsonl")
    ap.add_argument("--gen-url", default="http://127.0.0.1:8093/v1")
    ap.add_argument("--gen-model", default="qwen3-8b-chat")
    ap.add_argument("--gen-workers", type=int, default=64)
    ap.add_argument("--gen-max-tokens", type=int, default=400)
    ap.add_argument("--gen-extra", default='{"chat_template_kwargs": {"enable_thinking": false}}')
    ap.add_argument(
        "--generate-only",
        action="store_true",
        help="write the cache and stop (score in a second run, with the chat model's memory free)",
    )
    a = ap.parse_args(argv[:at])
    e = build_parser().parse_args(["eval", *argv[at + 1 :]])
    if e.cache_dir == "":
        e.cache_dir = None
    ks = tuple(int(x) for x in e.ks.split(","))

    data = Path(e.data)
    tools = load_tools(data / "tools.jsonl")
    queries = [q for q in load_queries(data / "queries.jsonl") if any(r > 0 for r in q.qrels.values())]
    if not e.with_inst:
        queries = [replace(q, instruction="") for q in queries]
    chat = make_chat(a)
    Path(a.cache).parent.mkdir(parents=True, exist_ok=True)
    responses = generate(chat, prompts(tools), a.cache, workers=a.gen_workers, system=SYSTEM)
    if a.generate_only:
        print(f"{len(responses)} answers in {a.cache}")
        return
    scorer = build_scorer(e)
    per_tool = [texts(t, scorer.tool_format(t), responses) for t in tools]
    n_desc = sum(x["describe"] != x["base"] for x in per_tool)
    n_rows = [len(x["rows"]) for x in per_tool]
    print(
        f"{scorer.name} on {data.name}: {len(tools)} tools, {len(queries)} queries; written by {chat.name}: "
        f"{n_desc} descriptions, {sum(n_rows)} requests ({sum(1 for n in n_rows if n)} tools, "
        f"{np.mean(n_rows):.2f} a tool)\n"
    )

    qv = scorer._vectors([scorer.query_format(q) for q in queries], "query", scorer.project_queries)
    ids = [t.id for t in tools]

    def doc_vectors(key: str) -> np.ndarray:
        return scorer._vectors([x[key] for x in per_tool], "document", scorer.project_tools)

    base = doc_vectors("base")
    rankings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    empty = np.zeros(0, dtype=np.int64), np.zeros((len(qv), 0))
    rankings["base"] = Ranker(qv, base, *empty).rank(None)
    rankings["describe"] = Ranker(qv, doc_vectors("describe"), *empty).rank(None)
    rankings["requests_concat"] = Ranker(qv, doc_vectors("requests_concat"), *empty).rank(None)
    row_text = [r for x in per_tool for r in x["rows"]]
    row_tool = np.array([n for n, x in enumerate(per_tool) for _ in x["rows"]], dtype=np.int64)
    if len(row_text):
        rv = scorer._vectors(row_text, "document", scorer.project_tools)
        rankings["requests_rows"] = rows_rank(qv, base, rv, row_tool)

    out: dict[str, Any] = {"tools": len(tools), "queries": len(queries), "writer": chat.name, "variants": {}}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    metrics = ["NDCG@10", f"Recall@{ks[0]}", "Recall@10", "Comprehensiveness@10"]
    print(f"| variant | {' | '.join(metrics)} | NDCG@10 cat-macro |")
    print("| --- | " + " | ".join("---:" for _ in metrics) + " | ---: |")
    for key, (idx, val) in rankings.items():
        rows, lists = evaluate(queries, idx, val, ids, ks, None)
        rep = report(queries, rows)
        out["variants"][key] = rep
        o, cm = rep["overall"], rep["category_macro"]
        cells = [f"{100 * o[m]:.2f}" if m in o else "—" for m in metrics]
        macro = f"{100 * cm['NDCG@10']:.2f}" if cm else "—"
        print(f"| {key} | {' | '.join(cells)} | {macro} |")
        if a.out:
            runs = [_runs_row(q, r, m) for q, r, m in zip(queries, lists, rows, strict=True)]
            name = scorer.name + ("" if key == "base" else f"+{key}[{chat.name}]")
            Path(f"{a.out}_{key}_runs.jsonl").write_text(
                json.dumps({"dataset": data.name, "scorer": name, "rows": runs}, indent=2)
            )
    if a.out:
        Path(f"{a.out}.json").write_text(json.dumps(out, indent=2))
        print(f"\nsaved {a.out}.json and {a.out}_<variant>_runs.jsonl")


if __name__ == "__main__":
    main()
