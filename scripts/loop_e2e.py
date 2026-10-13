"""Pre-release check of the loop on a real server: traffic over HTTP, then learn, calibrate and ab.

Starts ``callbus serve`` on an ingest dir with the release stack, plays generated requests through
``POST /v1/search`` the way an agent would, reports each top hit back through ``POST /v1/feedback``
(the client ran the tool itself, so nothing outside is touched), stops the server, and then runs the
loop over the log it wrote: ``learn --dry-run`` (how many pairs the log gave), ``calibrate
--from-log`` (what the band means on this traffic) and ``ab --dry-run``.

    uv run python scripts/loop_e2e.py --data data/w3 --requests data/dev_w3 --n 120 \
        --emb-url http://$GB10:8097/v1 --emb-model qwen3-emb-lora --out results/loop_e2e.json

The agent here calls whatever came first, so the log tells ``learn`` that the ranking was already
right and its gate publishes nothing: what this checks is that the loop runs end to end on real
HTTP traffic, not that learning gains anything (``scripts/learn_sim.sh`` measures that).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def post(url: str, token: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (loopback, our own server)
        return json.loads(r.read())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--requests", required=True, help="dir with queries.jsonl to play as traffic")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--emb-url", required=True)
    p.add_argument("--emb-model", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    data, token, port = Path(a.data), secrets.token_urlsafe(16), free_port()
    with open(Path(a.requests) / "queries.jsonl", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f][: a.n]
    base = f"http://127.0.0.1:{port}"
    serve = subprocess.Popen(
        [
            "uv",
            "run",
            "callbus",
            "serve",
            "--data",
            str(data),
            "--port",
            str(port),
            "--emb-url",
            a.emb_url,
            "--emb-model",
            a.emb_model,
            "--clm-ckpt",
            "none",
            "--host",
            "127.0.0.1",
        ],  # fmt: skip
        env={**os.environ, "CALLBUS_API_KEY": token},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out: dict = {"data": str(data), "requests": a.requests, "played": 0}
    try:
        ready = False
        for _ in range(240):  # a cold index embeds the catalogue first
            try:
                with urllib.request.urlopen(f"{base}/healthz", timeout=5) as r:  # noqa: S310
                    if json.loads(r.read()).get("ready"):
                        ready = True
                        break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            if serve.poll() is not None:
                print(f"serve exited: {serve.stdout.read()[-2000:] if serve.stdout else ''}")
                return 1
            time.sleep(2)
        if not ready:
            print("serve never became ready")
            return 1
        print(f"serve ready on {base}")

        searched = called = empty = 0
        confidences = []
        for q in queries:
            res = post(f"{base}/v1/search", token, {"query": q["text"]})
            searched += 1
            if res.get("confidence") is not None:
                confidences.append(res["confidence"])
            tools = res.get("tools") or []
            if not tools:
                empty += 1
                continue
            post(
                f"{base}/v1/feedback",
                token,
                {"name": tools[0]["name"], "outcome": "ok", "search_id": res["search_id"], "took_ms": 12},
            )
            called += 1
        out.update(played=searched, calls=called, empty_lists=empty, confidences=len(confidences))
        print(f"played {searched} requests, {called} reported a call, {empty} empty lists")
    finally:
        serve.send_signal(signal.SIGINT)
        try:
            serve.wait(timeout=30)
        except subprocess.TimeoutExpired:
            serve.kill()

    log = sorted((data / "usage").glob("usage-*.jsonl"))
    out["log_files"] = [f.name for f in log]
    print(f"\nusage log: {[f.name for f in log]}")

    def run(label: str, args: list[str]) -> str:
        print(f"\n=== {label} ===")
        r = subprocess.run(
            ["uv", "run", "callbus", *args], capture_output=True, text=True, timeout=1800, check=False
        )
        text = (r.stdout + r.stderr).strip()
        print("\n".join(text.splitlines()[-14:]))
        out.setdefault("steps", {})[label] = {"returncode": r.returncode, "tail": text[-2000:]}
        return text

    where = ["--data", str(data), "--emb-url", a.emb_url, "--emb-model", a.emb_model]
    common = [*where, "--clm-ckpt", "none"]  # learn has --init instead: its heads are what it trains
    run("learn --dry-run", ["learn", *where, "--init", "none", "--dry-run"])
    run("calibrate --from-log --dry-run", ["calibrate", *common, "--from-log", "--dry-run"])
    run("calibrate --from-log", ["calibrate", *common, "--from-log"])
    run("ab --dry-run", ["ab", "--data", str(data), "--dry-run"])
    run("search with the band", ["search", queries[0]["text"], *common, "--json", "--min-confidence", "0.05"])

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
