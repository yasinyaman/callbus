"""Does a catalogue's confidence band carry to requests it was not calibrated on? (backlog D2.14)

``toolrank calibrate`` keeps the best scores of requests written for a catalogue, and a search's
``confidence`` is where its own best score falls in that distribution. The acceptance question is
not how a calibration describes itself — on its own requests the 5% band turns away 5% by
construction — but what the band does to requests from somewhere else: a benchmark's real tasks,
written by people, against a calibration written by a model.

This script ranks a second set of requests with the same first stage, under the instruction the
calibration was made with, and prints per band the share it would turn away and the share of twins
(the same requests with their gold tools' sources hidden, so the catalogue has no answer) it would
catch. The rule in ``docs/plan/backlog.md`` (D2.14): the 5% band may turn away at most 7% of
answerable requests; the catch is reported, not a criterion.

uv run python scripts/confidence_accept.py --check data/livemcpbench_server -- \
    --data data/livemcpbench_server --scorer dense --emb-url http://127.0.0.1:8097/v1 \
    --emb-model qwen3-emb-lora --truncate 8192 --tool-format documentation \
    --query-format instruct_query --with-inst
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from toolrank.build import build_scorer
from toolrank.calibration import FILE, Calibration, load
from toolrank.calibration import measure as measure_scores
from toolrank.cli import build_parser
from toolrank.datasets.jsonl import load_queries, load_tools


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def pick(entries: list[Calibration], scorer: str) -> Calibration:
    """The entry for this first stage, else the only one there is (and say so)."""
    same = [e for e in entries if e.scorer == scorer]
    if same:
        return same[-1]
    if len(entries) == 1:
        print(f"note: the calibration is for {entries[0].scorer!r}, these flags build {scorer!r}")
        return entries[0]
    sys.exit(f"no calibration for {scorer!r}; the file has {[e.scorer for e in entries]}")


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: confidence_accept.py [flags] -- <toolrank eval flags>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="dir whose queries.jsonl is checked against the band")
    ap.add_argument("--calibration", default=None, help=f"calibration file (default: DATA/{FILE})")
    ap.add_argument("--bands", default="0.01,0.05,0.1", help="--min-confidence values to report")
    ap.add_argument("--out", default=None, help="write the numbers here as JSON")
    a = ap.parse_args(argv[:cut])
    e = build_parser().parse_args(["eval", *argv[cut + 1 :]])
    if e.cache_dir == "":
        e.cache_dir = None

    cal_path = Path(a.calibration) if a.calibration else Path(e.data) / FILE
    entries = load(cal_path)
    if not entries:
        sys.exit(f"{cal_path}: no calibration (run toolrank calibrate first)")

    tools = load_tools(Path(e.data) / "tools.jsonl")
    queries = load_queries(Path(a.check) / "queries.jsonl")
    scorer = build_scorer(e)
    scorer.index(tools)
    cal = pick(entries, getattr(scorer, "name", "?"))

    best, twins, recall, skipped = measure_scores(scorer, tools, queries, cal.instruction)
    if not best:
        sys.exit(f"{a.check}: no request could be scored ({skipped} skipped)")
    gone = f", {skipped} skipped (their tools are not in this catalogue)" if skipped else ""
    print(
        f"calibration: {len(cal.best)} requests from {cal.requests or '?'}, p5 "
        f"{cal.best[max(0, len(cal.best) * 5 // 100 - 1)]:.3f}, median {statistics.median(cal.best):.3f}"
    )
    print(
        f"checked: {len(best)} requests from {a.check}{gone}, p5 "
        f"{sorted(best)[max(0, len(best) * 5 // 100 - 1)]:.3f}, median {statistics.median(best):.3f}, "
        f"Recall@5 {recall * 100:.1f}%"
    )
    rows = []
    for band in _floats(a.bands):
        refused = sum(cal.confidence(s) < band for s in best) / len(best)
        caught = sum(cal.confidence(s) < band for s in twins) / len(twins) if twins else None
        rows.append({"band": band, "refused": refused, "caught": caught, "n": len(best)})
        kept = f", catches {caught * 100:.1f}% of the twins" if caught is not None else ""
        print(f"--min-confidence {band}: turns away {refused * 100:.1f}% of these requests{kept}")
    if a.out:
        Path(a.out).write_text(
            json.dumps(
                {
                    "calibration": {
                        "file": str(cal_path),
                        "scorer": cal.scorer,
                        "requests": cal.requests,
                        "n": len(cal.best),
                    },
                    "checked": {"dir": a.check, "n": len(best), "recall_at_5": recall, "skipped": skipped},
                    "bands": rows,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
