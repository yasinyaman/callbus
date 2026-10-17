import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from callbus.cli import main
from callbus.ingest import hub
from callbus.pins import Pins, doc_pin


def _record(source, doc):
    doc = {"server": source, **doc}
    return {"id": f"{source}/{doc['name']}", "doc": doc, "documentation": json.dumps(doc), "category": source}


TIME = {
    "name": "get_time",
    "description": "The time in a timezone.",
    "inputSchema": {"type": "object", "properties": {"tz": {"type": "string"}}},
}
PETS = {
    "name": "listPets",
    "description": "List pets.",
    "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
    "http": {
        "method": "GET",
        "path": "/pets",
        "base_url": "https://pets.example.com/v1",
        "args": {"limit": {"in": "query", "name": "limit"}},
    },
}


def _files(time_doc=TIME):
    lines = [_record("time", time_doc), _record("pets", PETS)]
    sources = {
        "time": {
            "kind": "mcp",
            "synced_at": "2026-10-16T04:09:53+00:00",
            "tools": 1,
            "transport": "stdio",
            "origin": "hub:io.github.x/time",
        },
        "pets": {
            "kind": "openapi",
            "synced_at": "2026-10-16T04:09:53+00:00",
            "tools": 1,
            "origin": "https://pets.example.com/openapi.json",
            "base_url": "https://pets.example.com/v1",
            "hub": "com.example/pets",
        },
        "later": {
            "kind": "mcp",
            "synced_at": "2026-10-16T04:09:53+00:00",
            "tools": 0,
            "transport": "http",
            "origin": "hub:io.github.x/later",
        },
    }
    return {"tools.jsonl": "".join(json.dumps(r) + "\n" for r in lines), "sources.json": json.dumps(sources)}


@pytest.fixture
def fake_hub():
    state = {"answer": (200, _files()), "seen": [], "auth": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server's name
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["seen"].append((self.path, self.headers["Content-Type"], body))
            state["auth"].append(self.headers.get("Authorization"))
            status, payload = state["answer"]
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_address[1]}"
    yield state
    srv.shutdown()


def test_ingest_hub_writes_the_hubs_tools_sources_and_pins(fake_hub, tmp_path, capsys):
    toolset = tmp_path / "toolset.yaml"
    toolset.write_text("name: demo\nservers:\n  - io.github.x/time\n")
    out = tmp_path / "tools"
    assert main(["ingest", "hub", fake_hub["url"] + "/", "--toolset", str(toolset), "--out", str(out)]) == 0
    path, ctype, body = fake_hub["seen"][0]
    assert (path, ctype, body) == ("/api/v1/deploy/ingest", "application/yaml", toolset.read_bytes())

    rows = [json.loads(line) for line in (out / "tools.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["pets/listPets", "time/get_time"]
    pets = rows[0]
    assert pets["doc"]["http"]["path"] == "/pets"  # the routing callbus calls the API with
    # indexed as callbus ingest would index it, not the hub's whole doc
    assert json.loads(pets["documentation"]) == {
        "server": "pets", "name": "listPets", "description": "List pets.", "inputSchema": PETS["inputSchema"],
    }  # fmt: skip
    manifest = json.loads((out / "sources.json").read_text())
    assert (
        manifest["pets"]["kind"] == "openapi"
        and manifest["pets"]["base_url"] == "https://pets.example.com/v1"
    )
    assert manifest["time"]["transport"] == "stdio" and manifest["time"]["origin"] == "hub:io.github.x/time"
    assert manifest["later"]["tools"] == 0

    pins = Pins.load(out / "pins.json")
    assert pins.servers["time"] == {"get_time": doc_pin(TIME)}
    assert pins.servers["later"] == {} and pins.snapshot == "2026-10-16T04:09:53+00:00"
    printed = capsys.readouterr().out
    assert "time  +1 ~0 -0 =0" in printed and "the hub has not listed later yet" in printed

    # the hub's tool changed: the listing syncs, the pins follow
    fake_hub["answer"] = (200, _files({**TIME, "description": "The time anywhere."}))
    assert main(["ingest", "hub", fake_hub["url"], "--toolset", str(toolset), "--out", str(out)]) == 0
    assert "time  +0 ~1 -0 =0" in capsys.readouterr().out
    assert Pins.load(out / "pins.json").servers["time"]["get_time"] != pins.servers["time"]["get_time"]


def test_a_source_the_sync_refused_keeps_its_old_pins(fake_hub, tmp_path):
    toolset = tmp_path / "toolset.yaml"
    toolset.write_text("servers: []\n")
    out = tmp_path / "tools"
    (out).mkdir()
    (out / "sources.json").write_text(json.dumps({"time": {"kind": "openapi", "tools": 0}}))
    (out / "pins.json").write_text(json.dumps({"servers": {"time": {"tools": {"old": "a" * 64}}}}))
    assert main(["ingest", "hub", fake_hub["url"], "--toolset", str(toolset), "--out", str(out)]) == 1
    pins = Pins.load(out / "pins.json")
    assert pins.servers["time"] == {"old": "a" * 64}  # time is an openapi source here: not replaced
    assert "listPets" in pins.servers["pets"]


def test_the_hubs_errors_come_back_readable(fake_hub, tmp_path):
    toolset = tmp_path / "toolset.yaml"
    toolset.write_text("servers: [nope]\n")
    fake_hub["answer"] = (400, {"detail": "nope: not in the hub"})
    with pytest.raises(SystemExit, match="HTTP 400: nope: not in the hub"):
        main(["ingest", "hub", fake_hub["url"], "--toolset", str(toolset), "--out", str(tmp_path / "o")])
    fake_hub["answer"] = (200, {"tools.jsonl": ""})
    with pytest.raises(SystemExit, match="expected"):
        main(["ingest", "hub", fake_hub["url"], "--toolset", str(toolset), "--out", str(tmp_path / "o")])
    with pytest.raises(ValueError, match="http"):
        hub.endpoint("hub.fx42.ai")
    assert (
        hub.endpoint("https://hub.fx42.ai/api/v1/deploy/ingest") == "https://hub.fx42.ai/api/v1/deploy/ingest"
    )
    stray = {"tools.jsonl": json.dumps(_record("ghost", TIME)) + "\n", "sources.json": "{}"}
    with pytest.raises(ValueError, match="does not name"):
        hub.listings(stray)


def test_the_hubs_token_comes_from_the_environment(fake_hub, tmp_path, monkeypatch):
    toolset = tmp_path / "toolset.yaml"
    toolset.write_text("servers: [io.github.x/time]\n")
    args = ["ingest", "hub", fake_hub["url"], "--toolset", str(toolset), "--out", str(tmp_path / "o")]
    monkeypatch.delenv(hub.TOKEN_ENV, raising=False)
    assert main(args) == 0 and fake_hub["auth"] == [None]  # none set, none sent
    monkeypatch.setenv(hub.TOKEN_ENV, "cbh_secret\n")
    assert main(args) == 0 and fake_hub["auth"][-1] == "Bearer cbh_secret"
    # the hub's API is closed without a token: the error says where one goes
    fake_hub["answer"] = (401, {"detail": "a token is required"})
    monkeypatch.delenv(hub.TOKEN_ENV)
    with pytest.raises(SystemExit, match=f"HTTP 401: a token is required \\(set {hub.TOKEN_ENV}"):
        main(args)
    monkeypatch.setenv(hub.TOKEN_ENV, "cbh_withdrawn")
    with pytest.raises(SystemExit, match=f"\\(the hub refused {hub.TOKEN_ENV}\\)"):
        main(args)
    monkeypatch.setenv(hub.TOKEN_ENV, "  ")  # blank is none
    with pytest.raises(SystemExit, match="set"):
        main(args)
    assert fake_hub["auth"][-1] is None


def test_the_token_goes_over_https_or_to_this_machine_only():
    with pytest.raises(ValueError, match="https only"):
        hub.fetch("http://hub.example.org", b"servers: []\n", token="cbh_secret")
