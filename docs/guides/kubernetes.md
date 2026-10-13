# Kubernetes

The chart in `deploy/helm/callbus` runs `callbus serve` with its embedding backbone. It is not
published to a chart repository yet: install it from a checkout.

```bash
helm install callbus deploy/helm/callbus -n callbus --create-namespace \
  --set auth.apiKey="$CALLBUS_API_KEY"
kubectl -n callbus port-forward svc/callbus 8765:8765
```

## The backbone

`embedding.mode` picks where the backbone runs:

| Mode | What runs | Needs |
| --- | --- | --- |
| `vllm` (default) | a vLLM pod next to callbus, weights on their own volume | a GPU node (`nvidia.com/gpu`) |
| `external` | nothing: `embedding.url` and `embedding.model` name your endpoint | an OpenAI-compatible `/v1/embeddings` |
| `bundled` | one pod with the `callbus-vllm` image | a GPU node, and the image built from `deploy/docker/Dockerfile.vllm` and pushed to your registry (`embedding.bundled.image`) |

`embedding.backbone` names the weights: callbus's LoRA-trained Qwen3-Embedding-8B by default
(`yasinyaman/callbus-emb-8b` at `v0.2`, served as `callbus-emb-v0.2`), or
`{repo: Qwen/Qwen3-Embedding-8B, revision: "", name: qwen3-emb}` for the base model with the packaged
heads. `embedding.profile` sets the precision: `fp8` (the default, quantized at load, about 8 GB,
served as `<name>-fp8`) or `bf16` (about 16 GB, `<name>`). In our runs FP8 stays within a point of
bf16. The embedding cache is keyed by the served name, so switching profiles
re-embeds the catalogue once. The first start of a vLLM pod downloads 16 GB; the probes allow 30
minutes for it. On k3s, and on other clusters where the NVIDIA runtime is not the default, set
`embedding.runtimeClassName=nvidia` (and install the NVIDIA device plugin, which advertises
`nvidia.com/gpu`).

## Your tools

`config` is the `callbus.json` of [Serve them to agents](serve.md#the-config-file): MCP servers
and OpenAPI base URLs and headers. An init container runs `callbus ingest mcp` on every start, so
`helm upgrade` with a new server rolls the pod and the server appears in the catalogue. Helm merges
your values into the chart's, so drop the example server explicitly (`mcpServers: {time: null}`).
Credentials go in a Secret listed under `envFrom` and are referred to as `${VAR}` in the config.
OpenAPI specs are ingested with `ingest.args`, or with `kubectl exec` into the pod: the server
re-reads the catalogue without a restart.

## Keys and tenants

At least one key is required, since the server listens on the pod network: `auth.apiKey` (or
`auth.apiKeySecret`) for one token, `auth.apiKeys` (or `auth.apiKeysSecret`, a Secret holding the
file) for [tenants](serve.md#tenants), each with its own sources and credentials. The Service's
names (`callbus`, `callbus.<namespace>.svc`, ...) are allowed as Host; add an ingress host to
`serve.allowedHosts`. Terminate TLS at the ingress.

## What it creates

One callbus Deployment (one replica, `Recreate`: the index snapshot, the usage log and the learned
heads live on one `ReadWriteOnce` volume), its Service, a ConfigMap, a Secret when the keys are
given inline, the data volume (and a weights volume for `vllm` and `bundled`). The volumes are kept
on `helm uninstall`: they hold the embedding cache and the usage log. `serve.args` passes further
flags (`--server-weight 0.2`, `--co-use 2`, `--allow-write`). Prometheus scrapes `/v1/metrics` with
a key that reaches every source ([Metrics](serve.md#metrics)).

## Trying it

`scripts/helm_smoke.py` installs the chart on any cluster without a GPU: a fake embedding server in
the cluster, `external` mode, then a search, a call to a stdio MCP server inside the pod, a refused
token, the metrics, a key limited to one source, an upgrade that adds a server, and an uninstall
that keeps the data volume. On a laptop, [kind](https://kind.sigs.k8s.io/) gives a cluster in
Docker:

```bash
kind create cluster --name callbus
uv run python scripts/helm_smoke.py                    # the published image
docker build -f deploy/docker/Dockerfile -t callbus:dev . && kind load docker-image callbus:dev --name callbus
uv run python scripts/helm_smoke.py --image callbus:dev --tenants
kind delete cluster --name callbus
```

Check the disk before you start: the kind node image is 1.3 GB, the loaded callbus image another
0.5–0.7 GB, and the pod writes its catalogue and index on top. A Docker install with no room left
fails in a way that looks like a chart bug — the ingest init container dies with
`OSError: [Errno 28] No space left on device` — and on Docker Desktop a disk image that fills up can
take the rest of your images and volumes with it when the VM recovers. `df -h /` and
`docker system df` first.

The GPU modes have been checked by rendering and by the API server's validation, not yet on a GPU
cluster.
