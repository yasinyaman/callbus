"""Claude with callbus as its tool search.

Every tool of the callbus catalogue goes to the Messages API with ``defer_loading: true``; Claude
gets one tool it can see, ``search_tools``, and callbus answers it with ``tool_reference`` blocks
that load the matching tools. Calls to them run through callbus (``POST /v1/call``). See
``callbus.integrations.anthropic`` for the rules the loop keeps.

    callbus serve --data data/mytools --config callbus.json          # in another shell
    uv run --extra anthropic python examples/anthropic_tool_reference.py "What time is it in Tokyo?"

``ANTHROPIC_API_KEY`` (and ``CALLBUS_API_KEY`` when the server has one) come from the environment
or from a ``.env`` file in the working directory, and are never printed. A call that could change
something — no ``readOnlyHint``, or an OpenAPI operation other than GET/HEAD — is asked about first
unless ``--yes``: a tool's output can steer the model. ``--builtin bm25`` swaps callbus's search
for the API's own BM25 tool search over the same deferred tools.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_env(path: str = ".env") -> None:
    """``KEY=value`` lines into the environment, for keys not set already."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for line in lines:
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    p = argparse.ArgumentParser(description="Claude + callbus tool search")
    p.add_argument("task")
    p.add_argument("--callbus", default=os.environ.get("CALLBUS_URL", "http://127.0.0.1:8765"))
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--servers", default="", help="comma-separated sources to offer (default: all)")
    p.add_argument("--builtin", choices=["bm25", "regex"], help="the API's own tool search instead")
    p.add_argument("--yes", action="store_true", help="run every call without asking")
    p.add_argument(
        "--inline",
        action="store_true",
        help="no catalogue: add found tools by value (beta inline-tools-2026-09-15)",
    )
    p.add_argument(
        "--prefetch", action="store_true", help="with --inline: search the task before the first turn"
    )
    a = p.parse_args()
    load_env()

    import anthropic

    from callbus.client import CallBusClient
    from callbus.integrations import anthropic as claude
    from callbus.integrations import read_only

    def approve(entry: dict, arguments: dict) -> bool:
        if a.yes or read_only(entry):
            return True
        answer = input(f"  run {entry['name']} {json.dumps(arguments, ensure_ascii=False)}? [y/N] ")
        return answer.strip().lower() in ("y", "yes")

    def show(kind: str, d: dict) -> None:
        if kind == "turn":
            print(f"turn {d['turn']}: {d['stop_reason']}")
        elif kind == "search":
            found = ", ".join(d.get("tools") or []) or d.get("error") or "nothing"
            print(f"  search {d.get('query')!r} -> {found}" + (f"  [{d['mode']}]" if d.get("mode") else ""))
        elif kind == "call":
            print(f"  call {d['tool']}: {d.get('outcome')}" + (f" ({d['ms']} ms)" if "ms" in d else ""))

    tr = CallBusClient(a.callbus, api_key=os.environ.get("CALLBUS_API_KEY"))
    servers = [s for s in a.servers.split(",") if s] or None
    box = claude.Toolbox(
        tr,
        servers=servers,
        builtin=a.builtin,
        approve=approve,
        on_event=show,
        inline=a.inline,
        prefetch=a.prefetch,
    )
    size = len(json.dumps(box.tools)) / 1e6
    print(f"{len(box.tools) - 1} tools deferred, {size:.1f} MB of definitions per request; model {a.model}")
    try:
        result = claude.run(
            anthropic.Anthropic(), box, a.task, model=a.model, max_tokens=a.max_tokens, max_turns=a.max_turns
        )
    except anthropic.APIError as e:
        print(f"Messages API error: {e}", file=sys.stderr)
        return 1
    print("\n" + (result.text or "(no text)"))
    print(f"\n{result.turns} requests, stop: {result.stop_reason}, usage: {result.usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
