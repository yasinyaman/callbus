# Serve them to agents

```bash
callbus serve --data tools/ --config callbus.json            # http://127.0.0.1:8765/mcp and /v1
callbus serve --data tools/ --config callbus.json --stdio    # for desktop clients
```

Needs the `[mcp]` extra. The agent sees two tools instead of hundreds:

- `search_tools(query, k?)` returns the matching tools with their input schemas (complete for the
  first three, shortened for the rest) and a `search_id`; a tool also carries `notes` when a hub's
  operators wrote some on it or its server ([ingest](ingest.md#from-a-hub));
- `call_tool(name, arguments, search_id?)` forwards the call to the tool's MCP server or OpenAPI
  operation. A call that fails validation returns the tool's full schema, so the agent can retry.

## The config file

`--config` tells `serve` how to reach the sources when a tool is called: an MCP client file
(`mcpServers`) plus an optional `openapi` section with each API's base URL and headers. `${VAR}` is
read from the environment.

```json
{
  "mcpServers": {"time": {"command": "uvx", "args": ["mcp-server-time"]}},
  "openapi": {"stripe": {"base_url": "https://api.stripe.com",
                         "headers": {"Authorization": "Bearer ${STRIPE_KEY}"}}}
}
```

An API's headers go only to its configured `base_url`, and redirects are not followed. OpenAPI
calls are GET and HEAD only unless you pass `--allow-write`.

## Pinned tools

```bash
callbus serve --data tools/ --config callbus.json --pins pins.json
```

A server can change what a tool says it does after you chose it. With `--pins` (the `pins.json` the
hub writes into a deployment, or the one `callbus ingest hub` writes), every call is checked before
it is sent: the tool as the catalogue holds it, which is what `search_tools` showed, and the tool as
the MCP server listed it when callbus connected must both still hash to their pin (name,
description and input schema). A source the file pins accepts only its pinned tools. A refused call
is an error result starting `pin_mismatch` and a `refused` call in the usage log. Sources the file
does not name are called as before. A server is listed once per connection, so a change it makes
while connected is caught at the next connection.

## Keys and hosts

On `127.0.0.1` (the default) no key is needed. Any other address needs one:

- `--api-key` or `CALLBUS_API_KEY`: one bearer token for everyone;
- `--api-keys keys.json`: one named token per client or team (`{"ci-agent": "${CI_AGENT_KEY}"}`);
  the name goes into the usage log as the tenant. A name is 1–64 letters, digits, `_`, `-` and `.`.

### Tenants

A named key can also be limited to some sources and carry its own credentials:

```json
{
  "ops": "${OPS_KEY}",
  "team-a": {
    "key": "${TEAM_A_KEY}",
    "sources": ["github", "time"],
    "headers": {"github": {"Authorization": "Bearer ${TEAM_A_GITHUB_TOKEN}"}},
    "env": {"time": {"TZ": "Europe/Istanbul"}}
  },
  "dashboard": {"key": "${DASHBOARD_KEY}", "scopes": ["search"]}
}
```

- `sources`: the key sees and calls only these sources' tools. Searches, `/v1/tools`, `/v1/rank` by
  id and calls leave the others out, and a tool outside them is answered like a tool that does not
  exist. Without `sources` the key reaches everything.
- `headers`: sent with this key's calls to that source on top of the config's: an OpenAPI source
  (to its configured `base_url` only) or a streamable HTTP MCP server.
- `env`: added to a stdio MCP server's environment.
- `scopes`: what the key may do, among `search` (searches, `/v1/tools`, `/v1/rank`), `call`
  (`call_tool`, `/v1/call`) and `feedback` (`/v1/feedback`); all three without it. A search-only key's
  MCP tool list has no `call_tool`, and a call it tries anyway is refused (403 over REST) and logged.

A source for which a key has `headers` or `env` gets a connection (for a stdio server, a process)
of that key's own, so one team's token never carries another team's call. The server refuses to
start when a key has credentials for a source it cannot send them to. Each key also has its own
co-use table and its own heads (`DATA/heads/tenants/<name>/`); the catalogue, the index and the
catalogue's embeddings are shared. What a key's requests leave in the caches (request embeddings,
the tools it hands `/v1/rank`, second-stage scores) is its own: another key's identical request is
not answered faster, so timing tells it nothing. `/v1/metrics` is server-wide, so a key limited by
`sources` cannot read it.

Requests are tidied before they are embedded, cached and logged: Unicode NFC, runs of spaces and
tabs as one space, at most one blank line in a row, nothing around them. Requests that differ only
in whitespace are therefore one cache entry.

The server checks the `Host` header, so a web page cannot reach it through a rebound domain. It
answers to its bind address and, when bound to `0.0.0.0`, to `localhost`. Add the names it is
reached by, such as a compose service or a proxy's host, with `--allowed-host NAME` or
`CALLBUS_ALLOWED_HOSTS=a,b`. A name without a port also matches requests with no port, as a
reverse proxy on 80 or 443 sends them.

The server speaks plain HTTP. To reach it from another machine, put a reverse proxy that terminates
TLS in front of it: without one the bearer token, the requests and the tools' results cross the
network in the clear.

## REST

The same port serves REST for platforms that search and call tools themselves:

| Endpoint | What it does |
| --- | --- |
| `POST /v1/search` | a search over the catalogue |
| `POST /v1/rank` | scores up to 200 tools you pass in, or catalogue ids |
| `POST /v1/call` | runs a catalogue tool |
| `GET /v1/tools`, `GET /v1/tools/{id}` | the catalogue (`?full=true` with schemas) |
| `GET /v1/metrics` | Prometheus metrics |
| `GET /openapi.json`, `GET /healthz` | the API description; readiness |

See the [REST reference](../reference/rest.md).

## Agents with only a shell

Some agents run commands but speak neither MCP nor REST. For them, the repository has a skill:
[`examples/skills/callbus`](https://github.com/yasinyaman/callbus/tree/main/examples/skills/callbus)
holds a `SKILL.md` and two standard-library scripts that wrap `/v1/search` and `/v1/call`. Copy the
folder to where your agent loads skills (for Claude Code, `~/.claude/skills/callbus`):

```bash
export CALLBUS_URL=http://127.0.0.1:8765 CALLBUS_API_KEY=...   # the running serve
python3 scripts/search.py "what time is it in Tokyo"              # JSON: search_id, tools
python3 scripts/call.py time/get_current_time '{"timezone": "Asia/Tokyo"}' --search-id <id>
```

- `call.py` exits with 1 when the tool reports an error and 2 when callbus refuses the call or is
  down. The agent reads the output either way.
- The `SKILL.md` stays short, since some harnesses put every skill's card into the system prompt.
  The tools' schemas come from the search.
- A call without `--search-id` is linked to the same client's latest search in the usage log.

## Many servers, several tools, nothing that fits

Four options for catalogues where plain ranking leaves something on the table. All are off by
default; the numbers are from the benchmarks ([Benchmarks](../benchmarks.md) explains the sets), and
each says which first stage it was measured on.

**`--server-weight 0.2`: a vote for the right server.** Each server is embedded once as a summary
(its name and tool names), and a tool's score becomes its own cosine plus 0.2 of the request's
cosine with its server. On catalogues of many servers this lifted the first hit on the base model
with the v0.1 heads: MCP-Zero (293 servers) top-1 79.9 → 81.0, LiveMCPBench NDCG@10 54.0 → 55.1,
with fewer tools returned at a higher recall. On the v0.2 backbone the same rule is a wash —
MCP-Zero top-1 88.6 unchanged and NDCG@10 93.7 → 93.8, LiveMCPBench 55.7 → 55.9 — so a stronger
first stage leaves it little to add. Choosing servers first and searching only those loses points
everywhere (the right server ranks first only 68–86% of the time), so the vote is soft. On a
catalogue whose "servers" are a few huge groups it does nothing useful (ToolRet's three categories:
−0.1 NDCG@10, −0.8 averaged by category on the v0.1 heads; +0.0 and −0.2 on v0.2), which is why it
is not the default.

**`--co-use 2`: tools that are called together.** The usage log knows which tools agents called
after the same request. With `--co-use N`, a result gains up to N tools that were called along with
one of its tools in at least two requests and at least half of that tool's requests; they come
last, marked `used_with`. On a simulated log this changed one list in twenty and raised the share
of requests that got *every* tool they needed by 0.6 points for 0.05 more tools per list; making
the ranked list longer buys a seventh of that per tool. The table is rebuilt from the last 30 daily
log files every five minutes, counts all API keys together, and is not applied to a request that
names its own `k`.

**`--rerank cross`: a second stage.** The first stage scores the request and each
tool apart; a second stage reads the request together with each of the top 20 tools' full
documentation (cut to 3,000 characters) and reorders them. How many tools a search returns still
comes from the first stage's cosines (adaptive K); the order comes from the second stage. It is
the largest single gain on catalogues the models never saw. Over the default v0.2 backbone, with the
local reranker: LiveMCPBench NDCG@10 55.7 → 61.2, MCP-Zero top-1 88.6 → 91.9, ToolRet 58.9 → 59.4.
Over the base model with the v0.1 heads: LiveMCPBench 54.0 → 62.7, MCP-Zero top-1 79.9 → 91.3,
ToolRet 54.0 → 58.1 ([the comparison](https://github.com/yasinyaman/callbus/blob/main/docs/reports/faz2-rerank.md)).

- `--rerank cross --rerank-emb-url http://HOST:PORT/v1`: Qwen3-Reranker-8B behind vLLM's score API
  (about 16 GB more GPU memory). One search is one call of 20 pairs: 1.7–2.5 s at p50 on a GB10
  shared with three other vLLM servers (0.3–0.6 s a query at 8 requests in flight; not measured on
  an idle GPU). `deploy/spark/compose.yaml`'s `rerank` profile shows the vLLM flags.

`--rerank-depth`, `--rerank-tool-format` and `--rerank-max-chars` change the setting; the defaults are
the one that measured best. Requests that name their own `k` are reranked too.

A reranker that is down, slow or busy does not cost a search its answer. Each call gets
`--rerank-timeout` seconds (10) and one retry, and at most `--rerank-workers` calls are in
flight at once from all requests together. That limit is for a local server
with one worker; raise it for a vLLM reranker that serves several clients. Waiting for a free slot
counts against the timeout. When the second stage fails, the search returns the first stage's order
with a note saying so. The reason goes to the server's log, and
`callbus_search_rerank_failed_total` counts these searches. `/v1/rank` always scores with the first
stage's cosines.

**`--cut-threshold T --cut-min 0`: say so when nothing fits.** By default a search returns at least
one tool. With a threshold and a minimum of zero, a request whose best score is below T gets an
empty list and a note telling the agent to answer without a tool or rephrase (add `--cut-margin 0.2`
to keep the usual cut above the threshold). Choose T on your own traffic: the scores of requests
that have a tool and of those that do not overlap, and their scale moves with the catalogue and the
way requests are written. At a T that turns away 1% of answerable requests, MCP-Zero catches a
quarter of the unanswerable ones; on LiveMCPBench no threshold is that cheap. The log keeps what a
turned-away search would have shown (`results`, with `shown: 0`), which is what to calibrate on.

**`callbus calibrate`, `confidence` and `--min-confidence`: a threshold that means the same
everywhere.** A raw score has a different scale on every catalogue. A calibration measures that
scale on your own catalogue. A chat model writes requests for your tools, each one answerable by
construction, and `calibrate` ranks them the way the server does. It keeps their best scores in
`DATA/calibration.json`:

```bash
callbus data gen-queries --data tools/ --out tools-requests/ --n 200
callbus calibrate --data tools/ --requests tools-requests/      # the same flags as search and serve
```

From then on, every search answered by that first stage carries a `confidence`. It is the share of
those answerable requests whose best score was at or below this one's, so 0.05 means only 5%
scored lower. An agent or a client can choose its own band from it.
`serve --min-confidence 0.05` turns away the requests below that share: an empty list with the
note above, about 5% of the requests the calibration was made of. A request that names its own `k`
is never turned away. `calibrate` also ranks each request with its tool's server hidden, and reports
the share of those "twins" that the band would catch: the unanswerable requests it can tell apart.
Calibration does not make the score separate them any better; it makes the threshold portable.

**Calibrate on requests that look like your traffic.** "5% of the requests the calibration was made
of" is the whole caveat: a band is only as representative as its requests. Measured on one
catalogue two ways — requests written by a chat model, one per tool, against the 94 multi-step tasks
people wrote for the same tools — the generated requests scored much higher (median best score 0.69
against 0.49), so their 5% band turned away **54%** of the real tasks. `--styles situation`, which
states a problem instead of naming the operation, narrowed it to 21%; tasks of two tools did not
help. Within one style the band is exact: calibrated on half of a set's requests, it turned away
5.3% of the other half. So treat `confidence` as "compared with the requests I was calibrated on",
keep `--min-confidence` off until you have requests that resemble what your agents send, and
re-calibrate when that changes ([the measurement](https://github.com/yasinyaman/callbus/blob/main/docs/reports/backlog-d214.md)).

**`--from-log`: calibrate on the traffic the server answered.** Once a server has run for a while,
its own log is the representative set:

```bash
callbus calibrate --data tools/ --from-log [--since 2026-10-01] [--tenant NAME]
```

It takes the searches whose call landed on a tool — the evidence that the request was answerable —
and recovers each request's vector from the embedding cache, the way `callbus learn` does; the log
itself keeps no request text. There is no Recall@5 to report (a log has no gold list, only the tool
the agent chose), and the twins are measured by hiding the called tool's source. It needs at least
50 such requests, so generated requests remain the way to start; swap to the log when the traffic is
there, and the band then means what it says.

A calibration holds for one backbone, set of heads and serving instruction. After any of them
changes (a new backbone, heads promoted by `callbus ab`, `--instruction`), run `calibrate` again.
Until then searches carry no confidence and nothing is turned away, and the server says so at
start. Adding a few tools does not move the distribution much.

## Metrics

`GET /v1/metrics` answers in Prometheus' text format, behind the same bearer token as the rest of
`/v1`:

```yaml
scrape_configs:
  - job_name: callbus
    metrics_path: /v1/metrics
    authorization: {credentials: "<CALLBUS_API_KEY>"}
    static_configs: [{targets: ["127.0.0.1:8765"]}]
```

| Metric | What it tells you |
| --- | --- |
| `callbus_searches_total{via,mode,arm}`, `callbus_search_duration_seconds` | traffic and ranking latency (the request's embedding included) |
| `callbus_search_tools_returned`, `callbus_search_empty_total`, `callbus_search_co_use_added_total` | how many tools a search hands over |
| `callbus_search_rerank_failed_total` | searches whose second stage failed or timed out (the first stage's order answered) |
| `callbus_search_confidence` | with a calibration: how sure searches are (a falling median: traffic the catalogue does not answer) |
| `callbus_search_returned_tokens_total`, `callbus_search_saved_tokens_total`, `callbus_catalog_tokens` | the token estimate (below) |
| `callbus_calls_total{kind,outcome,via}`, `callbus_call_duration_seconds` | calls forwarded and how they ended |
| `callbus_calls_linked_total{link}`, `callbus_called_tool_rank` | whether calls can be tied to a search, and where the called tool stood in it |
| `callbus_embedding_texts_total{kind,source}`, `callbus_embedding_tokens_total` | embedding-cache hits (`source="cache"`) against texts sent to the endpoint |
| `callbus_rerank_candidates_total{source}` | with `--rerank`: candidates the second stage scored from its cache against those its model scored |
| `callbus_catalog_tools`, `callbus_catalog_sources`, `callbus_index_ready`, `callbus_heads{arm}`, `callbus_build_info` | what is being served |

The token estimate answers "what did searching save over loading every tool?". A tool counts as
its name, description and input schema in JSON at four characters a token; a search *returned* the
tokens of the tools it handed over and *saved* the catalogue's total minus that. So
`saved / (saved + returned)` is the reduction: on a catalogue of 1,862 tools (910k tokens) a search
returns about 5k, a reduction above 99%. It is an estimate and a floor: schemas are counted in full
although later hits are shortened, and the first searches after a catalogue change claim nothing
while the new catalogue is being sized. The share of calls in the top five of their search is
`callbus_called_tool_rank_bucket{le="5"} / callbus_called_tool_rank_count`.

Labels never carry request text, tool arguments or API key names (`arm` says `tenant` for any
key's own heads), and the counters are server-wide: any valid key can read them. They count what
the usage log sees and keep counting under `--no-usage-log`; they start at zero with the process.

## The usage log

Every search and call goes to `DATA/usage/` (or `--usage-log DIR`), one JSON line per event in a
file per day, each call tied to the search that found its tool. It is what
[`callbus learn`](learn.md) trains the heads on.

The server also follows `DATA/heads` while it runs: `current.npz` there replaces the served heads,
`candidate.npz` takes `--candidate-share` of the requests, and `tenants/<name>/` holds the same for one
API key. See [Learn from the usage log](learn.md#trying-the-new-heads-on-live-traffic).

What it holds, and what it does not:

- Requests and call arguments are keyed digests (HMAC-SHA256 under `DATA/usage/.key`, a random
  32-byte key made on first use, mode 0600): repeats are recognisable, guesses are not, and the log
  alone reconstructs nothing. The request's embedding-cache key is a digest too (`emb_hmac`), which
  is how `learn` finds its vector without its text.
- Tool names, scores, outcomes, latencies and the server's own instruction are text; so are the
  tools a search added by co-use (`added`). A name the agent called that is no tool is what it typed,
  so it is a digest (`unknown:…`) like a request. An instruction sent with a request is a digest. The client (API key name, client app, remote address)
  is a digest; the session id and the tenant (the API key's name) are text.
- `--log-text` adds the request text, the error text of failed calls and unknown tool names; `--mask-pii` then replaces
  e-mail addresses, phone, card and IBAN numbers in them with tags (a pattern, not an understanding).
- `--no-usage-log` turns the log off.

For data-protection purposes (GDPR, KVKK): without `--log-text` the log holds no personal data of
the requests themselves, only digests under a key you hold; the `.key` file and the embedding cache
(`DATA/cache`, which holds the requests' vectors) are the parts to protect and to delete with the
log. To forget a period, delete its daily files; to forget one API key's traffic, filter its
`tenant` lines out. An embedding vector can be matched to text only by embedding a guess and
comparing, which needs the same model and the cache.
