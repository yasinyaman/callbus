# Index your tools

`callbus ingest` turns MCP servers and OpenAPI specs into an ingest directory (see
[Concepts](../concepts.md#the-ingest-directory)). Run it again whenever a source changes: it
prints what was added, changed and removed, and embeds only the new and changed tools.

## MCP servers

```bash
callbus ingest mcp --server time="uvx mcp-server-time" --out tools/            # stdio: NAME=COMMAND
callbus ingest mcp --server docs=https://mcp.example.com/mcp --out tools/       # streamable HTTP: NAME=URL
callbus ingest mcp --config ~/.config/claude/mcp.json --out tools/              # an MCP client's config file
callbus ingest mcp --config mcp.json --only github,jira --out tools/            # some of its servers
```

Needs the `[mcp]` extra. Servers are listed concurrently, each within `--timeout`; one that fails
keeps its previous tools and is reported. `${VAR}` in a config file is read from the environment,
and neither env vars nor headers are written to `sources.json`.

## OpenAPI specs

```bash
callbus ingest openapi https://petstore3.swagger.io/api/v3/openapi.json --name petstore --out tools/
callbus ingest openapi specs/billing.yaml --name billing --out tools/ --dry-run   # the diff only
```

OpenAPI 3.x, JSON or YAML (`[openapi]` for YAML), from a path or a URL. Each operation becomes a
tool: its parameters and request-body properties form one flat input schema, local `$ref`s are
inlined, and the HTTP routing is kept aside for `serve`. The tool text is shrunk schema-first to
6,000 characters, so a huge operation never loses its name or description.

## From a hub

```bash
export CALLBUS_HUB_TOKEN=cbh_...   # the hub's API token, from its operators
callbus ingest hub https://hub.fx42.ai --toolset toolset.yaml --out tools/
```

A hub (the catalogue at [hub.fx42.ai](https://hub.fx42.ai)) has already listed the tools of tens of
thousands of MCP servers and REST APIs. Name the ones you want in a `toolset.yaml`, the hub's own
format, and this takes the hub's listing of each instead of starting the servers yourself: one
source per server, an API with its spec and base URL. A server the hub has not listed yet arrives
as a source with 0 tools; `callbus ingest mcp` fills it later. The hub's API answers a token
its operators give out: callbus reads it from `CALLBUS_HUB_TOKEN` and sends it to the hub only,
over https (or to a hub on this machine). The hub sends no credentials back, and calls still go
to the servers you configure for `serve`; the hub's deploy page writes a `callbus.json` for the
same toolset.

It also writes `tools/pins.json`: a hash of every tool's name, description and input schema as the
hub saw them (`--pins-out` puts it elsewhere; other sources' pins stay). `callbus serve --pins` then
refuses a call to a tool that changed since (see [serving](serve.md#pinned-tools)).

When the hub's operators have written notes on these servers or their tools (when one applies,
what to do, how sure they are, and the run or page that showed it), it keeps them in
`tools/guidance.jsonl`, one note a line. Each sync replaces the notes of the sources it took, so a
note the hub withdrew goes; a hub that sends none (an older one, or no notes yet) leaves the file
out, and `callbus ingest drop` drops a source's notes with its tools. A note on one tool holds only
while that tool's name, description and input schema are those the hub listed: after an
`ingest mcp` that relists the tool differently, it is no longer shown.

`search_tools` (and `/v1/search`) hand the notes over with the tools a search finds, as `notes` on
each hit: the tool's own (up to five), then its server's (up to five), each with `cue`, `lesson`
and `confidence`. They are never in the tool definitions or any prompt the agent always carries, so
a catalogue with notes costs no more tokens per turn than one without.

## Schemas an agent API would refuse

Ingest checks each tool's input schema against JSON Schema 2020-12. Without the `[mcp]` extra (which
brings `jsonschema`), it checks only the common draft-04 and draft-07 habits. It also flags a root
that is not an object and a `$ref` that points nowhere. A failing tool is kept and counted in its
source's line, with the first few reasons printed. Claude's API refuses a whole request over one
such schema, so the platform records (`/v1/tools?full=true`, `/v1/search`) give that tool
`{"type": "object"}` and an `inputSchemaProblem` instead. MCP `search_tools` shows the schema as it
is.

## Removing a source

```bash
callbus ingest drop billing --out tools/
```

## Embedding ahead of time

With `--emb-url` (or `CALLBUS_EMB_URL`), ingest embeds the new and changed tools into
`tools/cache/` right away, with the model and truncation `search` and `serve` use, so their first
index is all cache hits:

```bash
callbus ingest mcp --config mcp.json --out tools/ --emb-url http://gpu-host:8091/v1
```

Use the same URL string `serve` will use: the cache is keyed by it.
