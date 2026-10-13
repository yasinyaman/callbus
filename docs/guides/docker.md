# Docker

Two images, both for `linux/amd64` and `linux/arm64`:

| Image | What it runs | Needs |
| --- | --- | --- |
| `ghcr.io/yasinyaman/callbus` | `callbus` (serve, ingest, search) with the packaged heads, Node.js and uv for stdio MCP servers | an embedding endpoint serving the backbone |
| `callbus-vllm`, built from `deploy/docker/Dockerfile.vllm` | the same, plus vLLM serving the backbone inside the container | an NVIDIA GPU (16 GB; NVIDIA driver 580 or newer) |

`deploy/docker/` has a compose file for each: `compose.yaml` runs the official vLLM image and the
published `callbus` image as two services, and `compose.bundle.yaml` builds `callbus-vllm` and runs
it alone. The bundled image is not published: it is the official vLLM image with callbus on top,
too large to build for two platforms on the free CI runners.

## Two services: vLLM and callbus

```bash
cd deploy/docker
cp .env.example .env                                   # set CALLBUS_API_KEY
docker compose run --rm callbus ingest mcp --config /config/callbus.json --out /data
docker compose up -d
```

```yaml
--8<-- "deploy/docker/compose.yaml:12:"
```

- `callbus.json` lists the MCP servers, as an MCP client's config file does; stdio servers run
  inside the callbus container (`uvx` and `npx` are there), HTTP ones anywhere it can reach.
- OpenAPI specs: `docker compose run --rm -v $PWD/specs:/specs:ro callbus ingest openapi
  /specs/billing.yaml --name billing --out /data`, and their base URLs and headers go in the
  `openapi` section of `callbus.json`.
- Re-run the ingest line after changing the list: the running server picks the new catalogue up.
- The catalogue, the embedding cache, the index and the usage log live in the `data` volume; the
  model in `models`; uvx and npx caches in `home`.
- callbus runs as uid 1000 in the image: what you mount for it (`callbus.json`, specs) must be
  readable to that uid, and a `/data` you bind-mount writable to it.

## One container: callbus-vllm

```bash
cd deploy/docker
cp .env.example .env
docker compose -f compose.bundle.yaml run --rm callbus ingest mcp --config /config/callbus.json --out /data
docker compose -f compose.bundle.yaml up -d
```

The first command builds the image from your checkout: it pulls the official vLLM image (about 10
GB) and fetches the released heads, checking their sha256. Every command that embeds (serve, ingest,
search) starts vLLM on `127.0.0.1` inside the container,
waits for it, then runs callbus; if either exits, the container exits, and the restart policy
brings both back. Other commands (`--version`, `heads`) run callbus alone.

Inside the container vLLM runs as root, for the GPU and the model volume. callbus and the stdio MCP
servers it starts never do: they run as the image's unprivileged `callbus` user or, when the data
directory is one you bind-mounted, as that directory's owner, so your files stay yours. The data
directory is `/data`, or the `--data` you give `serve` and `search`, or a command's `--out`; one that
Docker created for you (owned by root) is handed to the `callbus` user only at `/data`. Whatever you
mount for callbus (`callbus.json`, specs) must be readable to that user; callbus says which uid it
runs as when it cannot read a config. `docker compose exec callbus callbus …` runs as that user too.
With `--user`, vLLM and callbus both run as the user you name, and the model volume must be
writable to it. Under a rootless engine, a directory of yours is used as it is.

| Variable | Default | |
| --- | --- | --- |
| `CALLBUS_BACKBONE`, `CALLBUS_BACKBONE_REVISION` | `yasinyaman/callbus-emb-8b`, `v0.2` | the weights; `Qwen/Qwen3-Embedding-8B` for the base model (served as `qwen3-emb`, with the packaged heads) |
| `CALLBUS_FP8` | `1` | FP8 weights, served as `callbus-emb-v0.2-fp8`; `0` for bf16, served as `callbus-emb-v0.2` |
| `VLLM_GPU_MEMORY_UTILIZATION` | vLLM's | the share of GPU memory vLLM may take |
| `VLLM_EXTRA_ARGS` | | more `vllm serve` flags |

## FP8 or bf16

Both setups serve the backbone in FP8 (the bf16 weights quantized as vLLM loads them): half the
weight memory, and within a point of bf16 in our runs (see [Benchmarks](../benchmarks.md)).
FP8 and bf16 are served under different names (`callbus-emb-v0.2-fp8`, `callbus-emb-v0.2`), which keeps their
vectors apart in the embedding cache: switching re-embeds the catalogue once.

## Settings

| Variable | Used by | |
| --- | --- | --- |
| `CALLBUS_API_KEY` | serve | the bearer token; required on `0.0.0.0` |
| `CALLBUS_EMB_URL`, `CALLBUS_EMB_MODEL` | serve, ingest, search | the embedding endpoint and its served name |
| `CALLBUS_EMB_API_KEY` | serve, ingest, search | a bearer token for the embedding endpoint, if it needs one (`OPENAI_API_KEY` is sent only to `api.openai.com`) |
| `CALLBUS_ALLOWED_HOSTS` | serve | names the server answers to besides localhost (the compose service, a proxy's host) |
| `CALLBUS_HEADS` | serve, search | a heads file other than the packaged one |

The compose files publish the port on `127.0.0.1` only, and the server speaks plain HTTP: to reach
it from another machine, put a reverse proxy that terminates TLS in front rather than publishing
the port. Add the proxy's host name to `CALLBUS_ALLOWED_HOSTS`: a name without a port also matches
the requests a proxy on 80 or 443 forwards.

## GPU memory

The model takes about 8 GB in FP8 and 16 GB in bf16. vLLM reserves `VLLM_GPU_MEMORY_UTILIZATION` of
the GPU (0.9 by default) for the weights and its cache; on a GPU shared with other services, lower it
(0.2 on a 120 GB NVIDIA GB10).

## Building the images

From the repository root:

```bash
docker build -f deploy/docker/Dockerfile --build-context heads=dist/heads -t callbus .
docker build -f deploy/docker/Dockerfile.vllm -t callbus-vllm .
```

`--build-context heads=DIR` bakes in the released heads found in `DIR` (`callbus-heads-*.npz`).
Without it, the `callbus` image serves the backbone alone until `callbus heads pull`, and
`callbus-vllm` downloads the released heads while it builds (`--build-arg HEADS_PULL=0` skips that).
`scripts/container_smoke.py IMAGE` checks an image without a GPU, against a fake embedding server.
