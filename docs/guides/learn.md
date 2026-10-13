# Learn from the usage log

Every search and call `callbus serve` answers goes to its usage log: which tools a request
retrieved, which one the agent then called, and how that went. `callbus learn` turns that into
better heads for your catalogue, without ever seeing a request's text.

```bash
pip install "callbus[clm]"                 # torch, for training only; the result runs in numpy
callbus learn --data data/mytools          # the ingest dir callbus serve has been serving
callbus learn --data data/mytools --dry-run   # what the log yields, nothing trained (no torch)
```

## What it learns from

The log holds no request text (see [Serve them to agents](serve.md#the-usage-log)), only a keyed
digest of the request's embedding-cache key. That is enough: the request's backbone vector is in the
ingest dir's cache, found by digesting the cache's keys with the log's key, and the tools' vectors
come from the catalogue's text through the same cache. Nothing is sent to the embedding endpoint
unless a catalogue tool changed since it was served.

A request becomes a training pair when a search of it has calls linked to it:

| The agent | The pair gets |
| --- | --- |
| called a tool and it returned `ok` | a positive |
| called a tool and it returned `tool_error` | a weak positive (the tool was the one to try; `--strict` leaves it out) |
| was shown a tool and never called it | a hard negative |
| got `refused`, `protocol_error`, `timeout`, `unknown_tool` | nothing: that says nothing about the tool |

Searches of the same request merge; requests whose tools left the catalogue are dropped and
counted. Hard negatives the starting heads score like a positive are dropped as well
(`--neg-filter 0.95`): two tools that do the same job are not each other's negatives, a lesson from
training on mined negatives. Searches another backbone answered are skipped and counted too (the
log records the model): their vectors do not belong with this backbone's tools.

## How it decides

The newest 20% of the requests (`--dev-share`) are held out, by time, and never trained on. The
metric is the log's own: the share of those requests whose called tool ranks in the top 5 of the
whole catalogue. The starting heads compete as epoch 0, and a later epoch has to beat them there.
The start is what the server would serve (`--init` names another): the promoted
`DATA/heads/current.npz` when there is one (a tenant's own with `--tenant`), else the packaged
heads when they fit the backbone, else fresh heads that start as the identity — so on a headless
backbone like the default one, epoch 0 is the backbone's own score, not a random head's noise.

With `--dev DIR` (a benchmark-format set, as `callbus finetune` takes) the same heads are scored on
it every epoch; an epoch that wins on the log but loses more than `--max-drop` NDCG@10 points there
is not published either. The decision is one of:

- `published`: the heads are written as fp16 `.npz` (the `.pt` next to it) with what they were
  selected on in their `cfg`. By default that is `DATA/heads/candidate.npz`: a running server starts
  answering a share of the requests with it (see below). `--out FILE` writes them elsewhere, to serve
  with `CALLBUS_HEADS=<file>` or `--clm-ckpt <file>`.
- `no improvement`, `benchmark dropped`: nothing is written; the report says the numbers.
- `not enough pairs` (fewer than `--min-pairs`, 20): the log is too young.

The report goes to `results/learn_<name>.json`. `--since` and `--tenant` narrow the log to a period
or to one API key's traffic.

## Trying the new heads on live traffic

A held-out share of yesterday's requests is evidence; today's requests are the test. `callbus serve`
watches `DATA/heads` while it runs:

| File | What the server does |
| --- | --- |
| `current.npz` | serves these heads instead of the packaged ones (unless `--clm-ckpt` names some) |
| `candidate.npz` | answers `--candidate-share` (10%) of the requests with it, a session always on the same side (requests without one are split one by one) |
| `tenants/<name>/current.npz`, `candidate.npz` | the same, for the requests of one API key (`--api-keys`) |

A key with heads files of its own runs its own experiment (`learn --tenant`, `ab --tenant`). Every
other key's requests take part in the shared one, candidate share included, like requests without
a key; `ab` without `--tenant` leaves the keys with their own experiment out, and `learn` without
`--tenant` does not learn from searches their own heads answered.

A file that appears or changes is loaded in the background (its tools are re-projected from cached
vectors, nothing is embedded again); until then requests get the heads before it. The log records
which arm answered each search. `callbus ab` reads it back:

```bash
callbus ab --data data/mytools            # since the candidate appeared; --dry-run moves nothing
```

It prints one row per arm (the control and the candidate): its searches, how many led to a call, the
share whose called tool stood first, `mrr` with its standard error, and K (the tools a search handed
over).

`mrr` is the mean of 1/rank of the tool the agent called, over all of the arm's searches (a search
nobody acted on counts 0): it rises when the right tool stands higher and when more searches lead
to a call. With `--min-searches` (100) on both sides, the difference has to clear the noise: more
than one standard error of the difference (the arms answer different requests) and more than
`--margin` (0.01). A candidate that far above the control becomes `current.npz` (the heads it
replaces are kept as `previous-<stamp>.npz`), and one that far below is set aside as
`rejected-<stamp>.npz`. A tie within the noise is promoted only when the candidate hands over fewer
tools: its mean K lower by more than K's own standard error and a tenth of a tool. Any other tie
keeps running. `--promote` and `--rollback` decide by hand.

Run the two every night and the loop closes: yesterday's candidate is judged, then a new one is
learned. While a candidate is still being judged, `learn` leaves it alone (`--replace-candidate`
overwrites it, and its comparison starts over).

```bash
callbus ab --data data/mytools && \
  callbus learn --data data/mytools --replay data/toolret_train/pairs.jsonl
```

(`--replay` keeps the heads from forgetting what the backbone already ranked well; *Not forgetting*
below has the numbers and why the sample size matters.)

## Not forgetting

Heads trained on one catalogue's traffic can lose what the released ones knew. `--replay pairs.jsonl`
mixes general request-to-tool pairs into the training batches (`--replay-n`, 10,000 of them; ToolRet's
training pairs from `callbus data pull toolret-train` are what the released heads saw): a seeded
sample over the whole file, positives only, requests a `--dev` set asks about dropped. They are
embedded once into the same cache; the epoch is still picked on the log, and `--dev` still guards.

How much to replay is not a free choice. On the default backbone, learning from one catalogue's
traffic costs 1.5–3 NDCG@10 points on benchmarks the traffic never covers, and the guard only
protects the `--dev` set you name. 1,000 pairs changed none of that; 10,000 removed it — a
catalogue's own gain stayed (held-out 94.3 → 95.8) while the set that had lost 2.5 points came back
(58.9 → 56.4 → 59.3). Incidental gains on other sets shrink in exchange. Hence the default, and
hence `--replay` in the nightly command.

## What to expect

Measured on simulated traffic: a benchmark served as a catalogue, 70% of its queries played as
requests by an agent that calls the right tools when they are shown, and the other 30% never served
(`scripts/learn_sim.py` and `scripts/learn_sim.sh` in the repository; NDCG@10 on the held-out
queries, with the released heads as the start, on the base Qwen3-Embedding-8B). On the default v0.2
backbone, `learn` starts from identity heads instead — the last two rows — so its start is the
backbone's own ranking and there is less left to win.

| Catalogue | Requests logged | Usable pairs | Held-out NDCG@10 |
| --- | ---: | ---: | --- |
| ToolRet (44,453 tools) | 100 | 77 | nothing published |
| | 300 | 246 | 54.2 → 55.1 |
| | 1,000 | 801 | 54.2 → 55.3 |
| | 3,000 | 2,309 | 54.2 → 56.6 |
| | 5,573 | 4,313 | 54.2 → 57.2 |
| MCP-Zero (2,792 tools) | 300 | 288 | nothing published |
| | 1,954 | 1,857 | 89.2 → 92.0 |
| ToolRet, on the v0.2 backbone | 5,573 | 4,313 | 59.6 → 62.0 ± 0.5 |
| MCP-Zero, on the v0.2 backbone | 1,954 | 1,857 | 94.3 → 95.8 |

The ± is what the training seed alone moves: five runs of the same recipe on the same log and the
same split landed between 61.5 and 62.5 (cat-macro 55.9 to 57.7). Read your own before-and-after
with that in mind — on this setup a difference under about a point is the seed, not the learning.

What the numbers say:

- **A hundred requests are too few.** No epoch beat the starting heads on so small a dev set, and
  nothing was published; from a few hundred pairs on the gain is small, in the thousands it is three
  points.
- **The gain is on the tools your traffic asks for.** On ToolRet, held-out requests for a tool some
  logged request had also wanted went from 46.6 to 52.0; requests for tools the traffic never asked
  for stayed where they were (63.7 → 63.8), neither better nor worse. Where requests share one style
  (MCP-Zero's are all written the same way), new tools gain as well.
- **Little is forgotten.** Without `--replay`, the worst case on the other benchmarks was half a
  point (ToolRet, after learning from MCP-Zero traffic); the `--dev` guard watches for more.
- **Wrong calls need `--strict`.** When the agent picked a wrong tool one time in five and that call
  ended `tool_error`, the weak positives spoiled the training: nothing was published. With `--strict`
  the full gain came back from the remaining requests. If your agents' failed calls are mostly wrong
  picks rather than right tools with bad arguments, use it.
- **The A/B agrees.** The held-out requests played as the next period's traffic, half to the
  candidate: `mrr` 0.572 for the control, 0.593 for the candidate, and `callbus ab` promoted it.

A simulated agent is tidier than a real one, and a benchmark's requests repeat more than yours may:
take the table as the shape of the curve, not as a promise. Heads learned from one catalogue's traffic
are for that catalogue: the benchmark guard is there to catch the case where they stop being good at
anything else.
