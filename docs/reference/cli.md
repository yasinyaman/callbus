# CLI reference

Generated from `callbus`'s argument parser by `scripts/cli_reference.py`; `callbus <command>
--help` shows the same. Every command also takes `-h` / `--help`.

## `callbus eval`

index a tool set and score a query set

| Argument | Default | Description |
| --- | --- | --- |
| `--data DATA` | required | directory with tools.jsonl and queries.jsonl |
| `--scorer {bm25,dense,clm,cross}` | `bm25` |  |
| `--tool-format {documentation,name_desc,schema,example_call}` |  |  |
| `--query-format {plain,concat,instruct_query,clm}` |  |  |
| `--with-inst` |  | keep the task instruction (ToolRet 'w/ inst.' setting) |
| `--tasks TASKS` |  | comma-separated task filter |
| `--instruction INSTRUCTION` |  | use this instruction for every query (implies --with-inst) |
| `--limit LIMIT` |  | only the first N queries (smoke tests) |
| `--k K` | `100` | retrieval depth (ToolRet: 100) |
| `--ks KS` | `5,10,20` | reported cut-offs |
| `--batch BATCH` | `64` | queries per rank() call (latency is per query) |
| `--no-stem` |  | BM25 without Snowball stemming |
| `--emb-url EMB_URL` | `http://127.0.0.1:8090/v1` | OpenAI-compatible base URL |
| `--emb-model EMB_MODEL` | `qwen3-8b` |  |
| `--emb-batch EMB_BATCH` | `32` |  |
| `--truncate TRUNCATE` |  | vLLM truncate_prompt_tokens (CLM reference: 2048) |
| `--cache-dir CACHE_DIR` | `.cache/callbus` | embedding cache directory (default: .cache/callbus; '' = none) |
| `--clm-ckpt CLM_CKPT` |  |  |
| `--index INDEX` | `numpy` | vector index for dense/clm: numpy \| faiss \| pgvector |
| `--index-dir INDEX_DIR` |  | keep the index on disk here (default: in memory) |
| `--pg-dsn PG_DSN` |  | pgvector: Postgres DSN (default: $CALLBUS_PG_DSN) |
| `--pg-table PG_TABLE` | `callbus_eval` | pgvector: not a served catalogue's table (its rows go) |
| `--hybrid` |  | fuse BM25 (request without instruction) by RRF |
| `--rrf-k RRF_K` | `60` | RRF constant |
| `--rrf-depth RRF_DEPTH` | `100` | list depth taken from each arm |
| `--rrf-weight RRF_WEIGHT` | `1.0` | weight of the BM25 term (1 = plain RRF) |
| `--server-weight SERVER_WEIGHT` |  | dense, clm: tool score + this much of the request's cosine with the tool's server (its category) |
| `--cut-margin CUT_MARGIN` |  | keep tools within this cosine of the best |
| `--cut-threshold CUT_THRESHOLD` |  | keep tools at or above this cosine |
| `--cut-max CUT_MAX` | `10` |  |
| `--cut-min CUT_MIN` | `1` |  |
| `--rerank {dense,clm,cross}` |  | reorder the top --rerank-depth with a second dense / clm / cross-encoder scorer |
| `--rerank-depth RERANK_DEPTH` | `100` | tools per query reranked |
| `--rerank-emb-url RERANK_EMB_URL` |  | the second scorer's endpoint (default: --emb-url) |
| `--rerank-emb-model RERANK_EMB_MODEL` |  | (default: --emb-model) |
| `--rerank-truncate RERANK_TRUNCATE` |  | (default: --truncate) |
| `--rerank-clm-ckpt RERANK_CLM_CKPT` |  | --rerank clm: its heads (default: --clm-ckpt) |
| `--rerank-tool-format {documentation,name_desc,schema,example_call}` |  | (default: --tool-format) |
| `--rerank-query-format {plain,concat,instruct_query,clm}` |  | (default: --query-format) |
| `--rerank-max-chars RERANK_MAX_CHARS` |  | cut each candidate's text to this many characters |
| `--rerank-template {qwen3,bge}` |  | --rerank cross: the reranker's prompt format |
| `--cross-template {qwen3,bge}` |  | --scorer cross: the prompt format (default qwen3) |
| `--rerank-query-chars RERANK_QUERY_CHARS` |  | --rerank cross: characters of the request kept (6000) |
| `--cross-query-chars CROSS_QUERY_CHARS` |  | --scorer cross: the same (6000) |
| `--rerank-workers RERANK_WORKERS` | `1` | queries scored concurrently by the second scorer (cross: 8) |
| `--device DEVICE` |  | torch device for the CLM heads |
| `--out OUT` |  | results JSON path |
| `--runs-out RUNS_OUT` |  | per-query rows (top-20 ids, P@1, hit@5, NDCG@10) for compare --paired |

## `callbus compare`

markdown table across result files

| Argument | Default | Description |
| --- | --- | --- |
| `files` | required |  |
| `--paired` |  | two runs files (eval --runs-out): sign and paired permutation tests per query |
| `--data DATA` |  | with --paired: the dataset dir whose queries give each query's task and category, for per-category and cat-macro rows (runs rows carry them since 0.3.0; this is for older files) |
| `--subset SUBSET` |  | with --paired: pair only the queries of this dataset dir, e.g. one half of callbus data split |
| `--metrics METRICS` | `NDCG@10,Recall@10,Comprehensiveness@10` |  |
| `--cat-macro CAT_MACRO` |  | metrics to add as category macro-averages (the ToolRet paper's Average); default: the first --metrics entry, '' for none |

## `callbus data pull`

download a benchmark and convert it to JSONL

| Argument | Default | Description |
| --- | --- | --- |
| `dataset` | required |  |
| `--out OUT` |  | default: data/<dataset> (dashes as underscores) |
| `--tasks TASKS` |  |  |
| `--gen-url GEN_URL` | `http://127.0.0.1:8093/v1` | mcp-zero: OpenAI-compatible chat endpoint |
| `--gen-model GEN_MODEL` | `qwen3-8b-chat` | mcp-zero: model that writes the queries |
| `--gen-workers GEN_WORKERS` | `32` | mcp-zero: concurrent requests |
| `--gen-max-tokens GEN_MAX_TOKENS` | `256` |  |
| `--gen-extra GEN_EXTRA` | `{"chat_template_kwargs": {"enable_thinking": false}}` | JSON merged into every chat request (default turns off Qwen3 thinking on vLLM; '{}' for none) |

## `callbus data gen-queries`

Make a selection set that shares no query with a benchmark: sample tools evenly over the sources of an ingest (or benchmark) dir and let a chat model write one request per tool, in three styles. The result is a benchmark-format dir for eval, finetune --dev and learn --dev.

| Argument | Default | Description |
| --- | --- | --- |
| `--data DATA` | required | the catalogue: a dir with tools.jsonl |
| `--out OUT` | required | the set's directory (tools.jsonl, queries.jsonl) |
| `--n N` | `600` | tools sampled, one request each |
| `--seed SEED` |  |  |
| `--styles STYLES` | `task,step,goal` | request styles, used in turn: task, step, goal, situation (a problem stated without the operation: harder to match) |
| `--exclude DIR` |  | a benchmark dir whose queries the set must not repeat (repeatable) |
| `--tools-per-request K` | `1` | 2-4: tasks that need K related tools of one source, all of them gold (much harder; --styles is not used) |
| `--gen-url GEN_URL` | `http://127.0.0.1:8093/v1` | OpenAI-compatible chat endpoint |
| `--gen-model GEN_MODEL` | `qwen3-8b-chat` | the model that writes the requests |
| `--gen-workers GEN_WORKERS` | `32` | concurrent requests |
| `--gen-max-tokens GEN_MAX_TOKENS` | `256` |  |
| `--gen-extra GEN_EXTRA` | `{"chat_template_kwargs": {"enable_thinking": false}}` | JSON merged into every chat request (default turns off Qwen3 thinking on vLLM; '{}' for none) |

## `callbus data server-names`

copy a benchmark set with each tool's server name in its text (the _server sets)

| Argument | Default | Description |
| --- | --- | --- |
| `src` | required | a benchmark dir (tools.jsonl + queries.jsonl), e.g. data/mcp_zero |
| `--out OUT` |  | default: <src>_server |

## `callbus data split`

A seeded share of the queries to --out-a (pick checkpoints, epochs or settings on it), the rest to --out-b (score the choice there, never pick on it). Both keep the whole tools.jsonl.

| Argument | Default | Description |
| --- | --- | --- |
| `src` | required | a benchmark-format dir (tools.jsonl + queries.jsonl) |
| `--out-a OUT_A` | required | the --share half: the selection set |
| `--out-b OUT_B` | required | the rest: the confirmation set |
| `--share SHARE` | `0.5` |  |
| `--seed SEED` |  |  |

## `callbus data synth`

write a small synthetic tool set + queries

| Argument | Default | Description |
| --- | --- | --- |
| `--out OUT` | `data/synthetic` |  |
| `--n-tools N_TOOLS` | `300` |  |
| `--n-queries N_QUERIES` | `200` |  |
| `--seed SEED` | `7` |  |

## `callbus ingest mcp`

list tools from MCP servers (stdio or streamable HTTP)

| Argument | Default | Description |
| --- | --- | --- |
| `--server SERVER` |  | NAME=URL (streamable HTTP) or NAME=COMMAND (stdio) (repeatable) |
| `--config CONFIG` |  | MCP client config: {'mcpServers': ...} or VS Code {'servers': ...} (repeatable) |
| `--only ONLY` |  | comma-separated server names to take from the configs |
| `--timeout TIMEOUT` | `60.0` | seconds per server, start to last page |
| `--out OUT` | required | ingest directory (tools.jsonl + sources.json) |
| `--dry-run` |  | print the diff, write nothing |
| `--replace` |  | let a source replace one of another kind |
| `--allow-empty` |  | accept a listing of 0 tools |
| `--tool-format {documentation,name_desc,schema,example_call}` | `documentation` | text to embed |
| `--emb-url EMB_URL` |  | OpenAI-compatible base URL |
| `--emb-model EMB_MODEL` |  |  |
| `--emb-batch EMB_BATCH` | `32` |  |
| `--truncate TRUNCATE` |  | vLLM truncate_prompt_tokens (CLM reference: 2048) |
| `--cache-dir CACHE_DIR` |  | embedding cache directory (default: DIR/cache, next to tools.jsonl; '' = none) |

## `callbus ingest openapi`

one tool per operation of an OpenAPI 3.x spec (path or URL)

| Argument | Default | Description |
| --- | --- | --- |
| `spec` | required |  |
| `--name NAME` |  | source name (default: slug of info.title) |
| `--max-chars MAX_CHARS` | `6000` | text budget per tool |
| `--out OUT` | required | ingest directory (tools.jsonl + sources.json) |
| `--dry-run` |  | print the diff, write nothing |
| `--replace` |  | let a source replace one of another kind |
| `--allow-empty` |  | accept a listing of 0 tools |
| `--tool-format {documentation,name_desc,schema,example_call}` | `documentation` | text to embed |
| `--emb-url EMB_URL` |  | OpenAI-compatible base URL |
| `--emb-model EMB_MODEL` |  |  |
| `--emb-batch EMB_BATCH` | `32` |  |
| `--truncate TRUNCATE` |  | vLLM truncate_prompt_tokens (CLM reference: 2048) |
| `--cache-dir CACHE_DIR` |  | embedding cache directory (default: DIR/cache, next to tools.jsonl; '' = none) |

## `callbus ingest hub`

the tools a hub (hub.fx42.ai) has listed for the servers of a toolset.yaml, with their pins

| Argument | Default | Description |
| --- | --- | --- |
| `url` | required | the hub, e.g. https://hub.fx42.ai |
| `--toolset TOOLSET` | required | toolset.yaml: the servers to take (the hub's format) |
| `--timeout TIMEOUT` | `120.0` | seconds for the hub's answer |
| `--pins-out PINS_OUT` |  | where the pins go (default: OUT/pins.json; other sources' pins stay) |
| `--out OUT` | required | ingest directory (tools.jsonl + sources.json) |
| `--dry-run` |  | print the diff, write nothing |
| `--replace` |  | let a source replace one of another kind |
| `--allow-empty` |  | accept a listing of 0 tools |
| `--tool-format {documentation,name_desc,schema,example_call}` | `documentation` | text to embed |
| `--emb-url EMB_URL` |  | OpenAI-compatible base URL |
| `--emb-model EMB_MODEL` |  |  |
| `--emb-batch EMB_BATCH` | `32` |  |
| `--truncate TRUNCATE` |  | vLLM truncate_prompt_tokens (CLM reference: 2048) |
| `--cache-dir CACHE_DIR` |  | embedding cache directory (default: DIR/cache, next to tools.jsonl; '' = none) |

## `callbus ingest drop`

remove whole sources

| Argument | Default | Description |
| --- | --- | --- |
| `names` | required |  |
| `--out OUT` | required |  |

## `callbus search`

Rank the tools of an ingest dir for one request. Defaults: the callbus backbone on 127.0.0.1:8091 (callbus-emb-v0.2), no heads, adaptive K (margin 0.2, max 10), a persistent index in DIR/index.

| Argument | Default | Description |
| --- | --- | --- |
| `request` | required |  |
| `--json` |  | one JSON object instead of a table |
| `--data DATA` | required | ingest dir with tools.jsonl |
| `--instruction INSTRUCTION` |  | default: the heads' instruction |
| `--k K` |  | a fixed top-k instead of adaptive K |
| `--no-cut` |  | plain top --cut-max |
| `--clm-ckpt CLM_CKPT` |  | heads: .npz / .pt path, 'default' (downloads) or 'none' (no packaged heads; learned DATA/heads ones still serve) |
| `--tool-format {documentation,name_desc,schema,example_call}` |  |  |
| `--query-format {plain,concat,instruct_query,clm}` |  |  |
| `--index INDEX` | `numpy` | numpy \| faiss \| pgvector |
| `--index-dir INDEX_DIR` |  | default: DATA/index |
| `--pg-dsn PG_DSN` |  | pgvector: Postgres DSN (default: $CALLBUS_PG_DSN) |
| `--pg-table PG_TABLE` | `callbus_tools` |  |
| `--hybrid` |  | fuse BM25 by RRF (helps agent-written requests only) |
| `--rrf-k RRF_K` | `60` |  |
| `--rrf-depth RRF_DEPTH` | `100` |  |
| `--rrf-weight RRF_WEIGHT` | `1.0` |  |
| `--server-weight SERVER_WEIGHT` |  | add this much of the request's cosine with a tool's server to the tool's score (0 = off; try 0.2) |
| `--co-use N` |  | append up to N tools the usage log shows are called together with a tool in the list (0 = off) |
| `--no-stem` |  |  |
| `--device DEVICE` |  |  |
| `--cut-margin CUT_MARGIN` |  | keep tools within this cosine of the best |
| `--cut-threshold CUT_THRESHOLD` |  | keep tools at or above this cosine |
| `--cut-max CUT_MAX` | `10` |  |
| `--cut-min CUT_MIN` | `1` |  |
| `--min-confidence Q` |  | turn away a request less sure than this share of answerable ones (an empty list and a note; e.g. 0.05); needs DATA/calibration.json (callbus calibrate) |
| `--rerank {cross}` |  | cross: a local cross-encoder behind vLLM's score API (Qwen3-Reranker-8B) |
| `--rerank-depth RERANK_DEPTH` | `20` | tools reranked per request |
| `--rerank-emb-url RERANK_EMB_URL` |  | cross: the reranker's /v1 endpoint |
| `--rerank-emb-model RERANK_EMB_MODEL` | `qwen3-reranker` | cross: its served name |
| `--rerank-template {qwen3,bge}` | `qwen3` | cross: prompt format |
| `--rerank-tool-format {documentation,name_desc,schema,example_call}` | `documentation` | text per tool |
| `--rerank-max-chars RERANK_MAX_CHARS` | `3000` | characters kept per tool |
| `--rerank-query-chars RERANK_QUERY_CHARS` |  | cross: characters of the request (6000) |
| `--rerank-workers RERANK_WORKERS` | `1` | cross: scoring calls in flight at once, all requests together |
| `--rerank-timeout RERANK_TIMEOUT` | `10.0` | seconds per attempt (two attempts, a turn in the queue included); then the first stage's order answers |
| `--emb-url EMB_URL` |  | OpenAI-compatible base URL |
| `--emb-model EMB_MODEL` |  |  |
| `--emb-batch EMB_BATCH` | `32` |  |
| `--truncate TRUNCATE` |  | vLLM truncate_prompt_tokens (CLM reference: 2048) |
| `--cache-dir CACHE_DIR` |  | embedding cache directory (default: DIR/cache, next to tools.jsonl; '' = none) |

## `callbus calibrate`

Rank requests written for the catalogue's own tools (callbus data gen-queries --data DIR) the way search and serve do, with the same flags, and keep their best scores in DIR/calibration.json. Searches by that first stage, heads and instruction then carry a confidence (the share of these answerable requests that scored lower), and --min-confidence turns away requests below a share. Calibrate again after the backbone, the heads (callbus ab) or the instruction change.

| Argument | Default | Description |
| --- | --- | --- |
| `--requests REQUESTS` |  | a callbus data gen-queries dir over this catalogue |
| `--from-log` |  | calibrate on what serve answered instead: the requests whose call landed on a tool (their vectors come from the embedding cache; generated requests score higher than real traffic) |
| `--since SINCE` |  | --from-log: an ISO date or timestamp, searches from it on |
| `--tenant TENANT` |  | --from-log: one API key's searches |
| `--dry-run` |  | print the bands, write nothing |
| `--data DATA` | required | ingest dir with tools.jsonl |
| `--instruction INSTRUCTION` |  | default: the heads' instruction |
| `--k K` |  | a fixed top-k instead of adaptive K |
| `--no-cut` |  | plain top --cut-max |
| `--clm-ckpt CLM_CKPT` |  | heads: .npz / .pt path, 'default' (downloads) or 'none' (no packaged heads; learned DATA/heads ones still serve) |
| `--tool-format {documentation,name_desc,schema,example_call}` |  |  |
| `--query-format {plain,concat,instruct_query,clm}` |  |  |
| `--index INDEX` | `numpy` | numpy \| faiss \| pgvector |
| `--index-dir INDEX_DIR` |  | default: DATA/index |
| `--pg-dsn PG_DSN` |  | pgvector: Postgres DSN (default: $CALLBUS_PG_DSN) |
| `--pg-table PG_TABLE` | `callbus_tools` |  |
| `--hybrid` |  | fuse BM25 by RRF (helps agent-written requests only) |
| `--rrf-k RRF_K` | `60` |  |
| `--rrf-depth RRF_DEPTH` | `100` |  |
| `--rrf-weight RRF_WEIGHT` | `1.0` |  |
| `--server-weight SERVER_WEIGHT` |  | add this much of the request's cosine with a tool's server to the tool's score (0 = off; try 0.2) |
| `--co-use N` |  | append up to N tools the usage log shows are called together with a tool in the list (0 = off) |
| `--no-stem` |  |  |
| `--device DEVICE` |  |  |
| `--cut-margin CUT_MARGIN` |  | keep tools within this cosine of the best |
| `--cut-threshold CUT_THRESHOLD` |  | keep tools at or above this cosine |
| `--cut-max CUT_MAX` | `10` |  |
| `--cut-min CUT_MIN` | `1` |  |
| `--min-confidence Q` |  | turn away a request less sure than this share of answerable ones (an empty list and a note; e.g. 0.05); needs DATA/calibration.json (callbus calibrate) |
| `--rerank {cross}` |  | cross: a local cross-encoder behind vLLM's score API (Qwen3-Reranker-8B) |
| `--rerank-depth RERANK_DEPTH` | `20` | tools reranked per request |
| `--rerank-emb-url RERANK_EMB_URL` |  | cross: the reranker's /v1 endpoint |
| `--rerank-emb-model RERANK_EMB_MODEL` | `qwen3-reranker` | cross: its served name |
| `--rerank-template {qwen3,bge}` | `qwen3` | cross: prompt format |
| `--rerank-tool-format {documentation,name_desc,schema,example_call}` | `documentation` | text per tool |
| `--rerank-max-chars RERANK_MAX_CHARS` | `3000` | characters kept per tool |
| `--rerank-query-chars RERANK_QUERY_CHARS` |  | cross: characters of the request (6000) |
| `--rerank-workers RERANK_WORKERS` | `1` | cross: scoring calls in flight at once, all requests together |
| `--rerank-timeout RERANK_TIMEOUT` | `10.0` | seconds per attempt (two attempts, a turn in the queue included); then the first stage's order answers |
| `--emb-url EMB_URL` |  | OpenAI-compatible base URL |
| `--emb-model EMB_MODEL` |  |  |
| `--emb-batch EMB_BATCH` | `32` |  |
| `--truncate TRUNCATE` |  | vLLM truncate_prompt_tokens (CLM reference: 2048) |
| `--cache-dir CACHE_DIR` |  | embedding cache directory (default: DIR/cache, next to tools.jsonl; '' = none) |

## `callbus serve`

One MCP server with two tools, search_tools and call_tool, in front of every tool of an ingest dir; calls go to the MCP servers of --config/--server and to OpenAPI operations (GET/HEAD unless --allow-write). HTTP by default (MCP at /mcp), or --stdio for desktop clients.

| Argument | Default | Description |
| --- | --- | --- |
| `--data DATA` | required | ingest dir with tools.jsonl |
| `--instruction INSTRUCTION` |  | default: the heads' instruction |
| `--k K` |  | a fixed top-k instead of adaptive K |
| `--no-cut` |  | plain top --cut-max |
| `--clm-ckpt CLM_CKPT` |  | heads: .npz / .pt path, 'default' (downloads) or 'none' (no packaged heads; learned DATA/heads ones still serve) |
| `--tool-format {documentation,name_desc,schema,example_call}` |  |  |
| `--query-format {plain,concat,instruct_query,clm}` |  |  |
| `--index INDEX` | `numpy` | numpy \| faiss \| pgvector |
| `--index-dir INDEX_DIR` |  | default: DATA/index |
| `--pg-dsn PG_DSN` |  | pgvector: Postgres DSN (default: $CALLBUS_PG_DSN) |
| `--pg-table PG_TABLE` | `callbus_tools` |  |
| `--hybrid` |  | fuse BM25 by RRF (helps agent-written requests only) |
| `--rrf-k RRF_K` | `60` |  |
| `--rrf-depth RRF_DEPTH` | `100` |  |
| `--rrf-weight RRF_WEIGHT` | `1.0` |  |
| `--server-weight SERVER_WEIGHT` |  | add this much of the request's cosine with a tool's server to the tool's score (0 = off; try 0.2) |
| `--co-use N` |  | append up to N tools the usage log shows are called together with a tool in the list (0 = off) |
| `--no-stem` |  |  |
| `--device DEVICE` |  |  |
| `--cut-margin CUT_MARGIN` |  | keep tools within this cosine of the best |
| `--cut-threshold CUT_THRESHOLD` |  | keep tools at or above this cosine |
| `--cut-max CUT_MAX` | `10` |  |
| `--cut-min CUT_MIN` | `1` |  |
| `--min-confidence Q` |  | turn away a request less sure than this share of answerable ones (an empty list and a note; e.g. 0.05); needs DATA/calibration.json (callbus calibrate) |
| `--rerank {cross}` |  | cross: a local cross-encoder behind vLLM's score API (Qwen3-Reranker-8B) |
| `--rerank-depth RERANK_DEPTH` | `20` | tools reranked per request |
| `--rerank-emb-url RERANK_EMB_URL` |  | cross: the reranker's /v1 endpoint |
| `--rerank-emb-model RERANK_EMB_MODEL` | `qwen3-reranker` | cross: its served name |
| `--rerank-template {qwen3,bge}` | `qwen3` | cross: prompt format |
| `--rerank-tool-format {documentation,name_desc,schema,example_call}` | `documentation` | text per tool |
| `--rerank-max-chars RERANK_MAX_CHARS` | `3000` | characters kept per tool |
| `--rerank-query-chars RERANK_QUERY_CHARS` |  | cross: characters of the request (6000) |
| `--rerank-workers RERANK_WORKERS` | `1` | cross: scoring calls in flight at once, all requests together |
| `--rerank-timeout RERANK_TIMEOUT` | `10.0` | seconds per attempt (two attempts, a turn in the queue included); then the first stage's order answers |
| `--emb-url EMB_URL` |  | OpenAI-compatible base URL |
| `--emb-model EMB_MODEL` |  |  |
| `--emb-batch EMB_BATCH` | `32` |  |
| `--truncate TRUNCATE` |  | vLLM truncate_prompt_tokens (CLM reference: 2048) |
| `--cache-dir CACHE_DIR` |  | embedding cache directory (default: DIR/cache, next to tools.jsonl; '' = none) |
| `--config CONFIG` |  | MCP client config (+ an optional 'openapi' section) (repeatable) |
| `--server SERVER` |  | NAME=URL (streamable HTTP) or NAME=COMMAND (stdio) (repeatable) |
| `--stdio` |  | speak MCP over stdin/stdout instead of HTTP |
| `--host HOST` | `127.0.0.1` |  |
| `--port PORT` | `8765` |  |
| `--api-key API_KEY` |  | bearer token for /mcp and /v1 (default: $CALLBUS_API_KEY) |
| `--api-keys FILE` |  | JSON {name: key}, one bearer token per client or team; the name is logged as the tenant |
| `--allowed-host ALLOWED_HOST` |  | extra Host header value to accept, e.g. the name a proxy or container network uses (also $CALLBUS_ALLOWED_HOSTS, comma-separated) (repeatable) |
| `--allow-write` |  | let call_tool send non-GET OpenAPI requests |
| `--pins FILE` |  | pins.json (from the hub, or callbus ingest hub): refuse a call to a tool whose description or input schema changed since then (pin_mismatch), and to a pinned source's unpinned tools |
| `--timeout TIMEOUT` | `60.0` | seconds per backend call and connection |
| `--usage-log USAGE_LOG` |  | usage log directory (default: DATA/usage) |
| `--no-usage-log` |  |  |
| `--candidate-share CANDIDATE_SHARE` | `0.1` | share of requests answered with DATA/heads/candidate.npz when there is one (sticky per session; requests without one are split one by one) |
| `--log-text` |  | also log request and error text |
| `--mask-pii` |  | with --log-text: mask e-mail, phone, card and IBAN numbers |

## `callbus finetune`

Embed the pairs once (only what the cache lacks), train heads on the frozen backbone's vectors, pick the epoch on --dev (a benchmark-format set that is not reported), then run callbus eval with the saved heads on --dev and every --eval set. Defaults: the setting of the released heads (skip heads, lr 1e-5, batch 512, 5 epochs, in-batch negatives only) and the encoder of the released heads (the base model as qwen3-emb on 8091, documentation + instruct_query, truncate 8192), whose heads search and serve take with --emb-model qwen3-emb.

| Argument | Default | Description |
| --- | --- | --- |
| `--data DATA` | required | pairs.jsonl (data pull toolret-train, or your own) |
| `--dev DEV` | required | benchmark-format dir the epoch is picked on; not reported |
| `--eval EVAL` |  | benchmark-format dir to evaluate (repeatable) |
| `--out OUT` | required | the heads, a torch .pt |
| `--npz NPZ` |  | also export the packaged fp16 .npz |
| `--init-ckpt INIT_CKPT` |  | .pt, .npz or 'default' (the packaged heads); default: fresh skip heads |
| `--name NAME` |  | the report's name (default: --out's stem) |
| `--results RESULTS` | `results` | where the report and eval JSONs go |
| `--report REPORT` |  | the report's path (default: RESULTS/finetune_<name>.json) |
| `--select {ndcg10,ndcg10-cat}` | `ndcg10` | dev metric |
| `--curve` |  | score the eval sets every epoch too (reported only) |
| `--embed-only` |  | fill the embedding cache and stop (no torch) |
| `--tool-format {documentation,name_desc,schema,example_call}` | `documentation` |  |
| `--query-format {plain,concat,instruct_query,clm}` | `instruct_query` |  |
| `--instruction INSTRUCTION` |  | for pairs without one (default: the serving one) |
| `--keep-bare` |  | leave pairs without an instruction bare |
| `--backbone BACKBONE` |  | recorded in the heads' cfg (default: what --emb-model names) |
| `--n-train N_TRAIN` |  | training pairs (0 = all) |
| `--n-val N_VAL` |  | held-out training pairs: a diagnostic, never selected on |
| `--seed SEED` |  |  |
| `--data-seed DATA_SEED` |  | seed of the train/val split, so the same pairs split alike across training seeds (default: --seed) |
| `--epochs EPOCHS` | `5` |  |
| `--batch BATCH` | `512` |  |
| `--lr LR` | `1e-05` | skip heads collapse at 3e-4 and above |
| `--neg NEG` |  | mined negatives per pair (ToolRet's cost up to 10 NDCG points) |
| `--neg-filter NEG_FILTER` |  |  |
| `--weight-decay WEIGHT_DECAY` | `0.01` |  |
| `--warmup WARMUP` | `0.05` |  |
| `--width WIDTH` |  | fresh heads' hidden width (default 1536) |
| `--depth DEPTH` |  | fresh heads' depth (default 3) |
| `--no-skip` |  | fresh heads without the x + MLP(x) skip |
| `--freeze-action` |  | train the state head only |
| `--device DEVICE` |  |  |
| `--emb-url EMB_URL` |  | OpenAI-compatible base URL |
| `--emb-model EMB_MODEL` |  |  |
| `--emb-batch EMB_BATCH` | `128` |  |
| `--truncate TRUNCATE` | `8192` | vLLM truncate_prompt_tokens (CLM reference: 2048) |
| `--cache-dir CACHE_DIR` | `.cache/callbus` | embedding cache directory (default: .cache/callbus; '' = none) |

## `callbus learn`

From an ingest dir callbus serve has served: the log's searches and calls become request -> tool pairs (a call that ended ok is a positive, a tool_error a weak one, tools shown but not called are hard negatives), the requests' vectors come from DATA/cache through the log's key, never their text, and the heads train from the served ones. The newest 20% of requests are the dev set (Recall@5 of the called tool over the catalogue); the heads are written only when they beat the starting ones there, and, with --dev, do not fall on a benchmark set.

| Argument | Default | Description |
| --- | --- | --- |
| `--data DATA` | required | the ingest dir: tools.jsonl, cache/, usage/ |
| `--out OUT` |  | the .npz to write (default: DATA/heads/candidate.npz, or tenants/<name>/ with --tenant: a running server gives it a share of the requests) |
| `--replace-candidate` |  | train even though a candidate is still being judged |
| `--replay REPLAY` |  | general pairs.jsonl mixed into training, against forgetting |
| `--replay-n REPLAY_N` | `10000` | how many of --replay's pairs (1,000 did not stop the forgetting, 10,000 did) |
| `--dev DEV` |  | benchmark-format dir scored alongside: a guard against forgetting |
| `--init INIT` | `default` | heads to start from: default (the served ones: DATA/heads/current.npz when there, else the packaged ones when they fit the backbone, else fresh identity heads), a path, or none |
| `--since SINCE` |  | only searches from this ISO date or timestamp on |
| `--tenant TENANT` |  | only one API key's searches (its name) |
| `--strict` |  | a tool_error call is not a (weak) positive |
| `--min-pairs MIN_PAIRS` | `20` | fewer usable requests: nothing is trained |
| `--dev-share DEV_SHARE` | `0.2` | share of the newest requests held out |
| `--max-drop MAX_DROP` | `0.5` | NDCG@10 points the --dev set may lose |
| `--dry-run` |  | mine and match the vectors, train nothing (no torch) |
| `--name NAME` |  | the report's name (default: the output's stem) |
| `--results RESULTS` | `results` | where the report goes |
| `--report REPORT` |  | the report's path (default: RESULTS/learn_<name>.json) |
| `--tool-format {documentation,name_desc,schema,example_call}` | `documentation` |  |
| `--query-format {plain,concat,instruct_query,clm}` | `instruct_query` | for --dev's queries |
| `--backbone BACKBONE` |  | recorded in the heads' cfg (default: what --emb-model names) |
| `--epochs EPOCHS` | `3` |  |
| `--batch BATCH` | `256` |  |
| `--lr LR` | `1e-05` |  |
| `--neg NEG` | `5` | shown-but-not-called tools per request in a batch |
| `--neg-filter NEG_FILTER` | `0.95` | drop negatives the start scores like a positive |
| `--weight-decay WEIGHT_DECAY` | `0.01` |  |
| `--warmup WARMUP` | `0.05` |  |
| `--seed SEED` |  |  |
| `--data-seed DATA_SEED` |  | seed of the replay sample, so replay picks alike across training seeds (default: --seed) |
| `--device DEVICE` |  |  |
| `--emb-url EMB_URL` |  | OpenAI-compatible base URL |
| `--emb-model EMB_MODEL` |  |  |
| `--emb-batch EMB_BATCH` | `128` |  |
| `--truncate TRUNCATE` |  | vLLM truncate_prompt_tokens (CLM reference: 2048) |
| `--cache-dir CACHE_DIR` |  | embedding cache directory (default: DIR/cache, next to tools.jsonl; '' = none) |

## `callbus ab`

Since DATA/heads/candidate.npz appeared, a share of the requests was answered with it. Per arm: the searches, how many led to a call, and how high the called tool stood (mrr: the mean of 1/rank over all the arm's searches). The candidate becomes current.npz when its mrr is --margin above the control's with --min-searches on both sides, is set aside when it is that much below, and keeps running otherwise. A running server follows the files.

| Argument | Default | Description |
| --- | --- | --- |
| `--data DATA` | required | the ingest dir: usage/ and heads/ |
| `--tenant TENANT` |  | one API key's heads (DATA/heads/tenants/<name>) |
| `--min-searches MIN_SEARCHES` | `100` | per arm, before anything is decided |
| `--margin MARGIN` | `0.01` | the mrr difference that decides |
| `--since SINCE` |  | ISO timestamp (default: when the candidate appeared) |
| `--dry-run` |  | say the decision, move nothing |
| `--promote` |  | promote now |
| `--rollback` |  | roll back now |
| `--results RESULTS` | `results` | where the report goes |

## `callbus heads export`

a torch .pt checkpoint -> the .npz that runs without torch

| Argument | Default | Description |
| --- | --- | --- |
| `src` | required |  |
| `dst` | required |  |
| `--dtype {float16,float32}` | `float16` |  |
| `--backbone BACKBONE` |  | serving defaults stored in the checkpoint cfg |
| `--tool-format TOOL_FORMAT` |  |  |
| `--query-format QUERY_FORMAT` |  |  |
| `--truncate TRUNCATE` |  |  |
| `--instruction INSTRUCTION` |  |  |

## `callbus heads pull`

download the packaged heads into the cache (sha256-checked); search and serve use them on the backbone they were trained on (--emb-model qwen3-emb or qwen3-emb-fp8)

| Argument | Default | Description |
| --- | --- | --- |
| `--url URL` |  | a mirror of the file (default: CALLBUS_HEADS_URL, then the release) |

## `callbus formats`

list tool / query text formats
