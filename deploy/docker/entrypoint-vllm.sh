#!/usr/bin/env bash
# The callbus-vllm image's entrypoint. For commands that embed (serve, ingest, search, eval,
# finetune) it starts vLLM with the embedding backbone on 127.0.0.1, waits until it answers, then
# runs callbus with the container's arguments; when either process exits, the other is stopped and
# the container exits with that status, so the runtime's restart policy takes over. Other commands
# (--version, heads, formats) run callbus alone.
#
# vLLM runs as the container's user (root by default: the GPU and /models). `callbus` in this image
# is a wrapper that never runs it as root (as-callbus.sh: the owner of /data, else the image's
# `callbus` user), so the stdio MCP servers callbus starts are not root either.
#
#   CALLBUS_BACKBONE=REPO           the weights (default: callbus's LoRA-trained Qwen3-Embedding-8B,
#   CALLBUS_BACKBONE_REVISION=TAG   yasinyaman/callbus-emb-8b at v0.2; Qwen/Qwen3-Embedding-8B for
#                                    the base model, which the packaged heads go with)
#   CALLBUS_FP8=1|0                 FP8 weights (quantized at load, served as <name>-fp8) or bf16
#                                    (<name>): the names keep the two apart in the embedding cache;
#                                    <name> is callbus-emb-v0.2, qwen3-emb for the base model, else
#                                    the repo id with / as -
#   VLLM_GPU_MEMORY_UTILIZATION=0.9  the share of GPU memory vLLM may take
#   VLLM_EXTRA_ARGS="..."            more `vllm serve` flags
#   CALLBUS_API_KEY                 required by `serve` on 0.0.0.0
#   CALLBUS_ENTRYPOINT_DRY_RUN=1    print the two command lines and exit
set -euo pipefail

case "${1:-}" in
  serve | ingest | search | eval | finetune) ;;
  *) exec callbus "$@" ;;
esac

default_repo=yasinyaman/callbus-emb-8b default_revision=v0.2 # build.BACKBONE_REPO / _REVISION
model=${CALLBUS_BACKBONE:-$default_repo}
case "$model" in
  "$default_repo") name=callbus-emb-$default_revision revision=${CALLBUS_BACKBONE_REVISION:-$default_revision} ;;
  Qwen/Qwen3-Embedding-8B) name=qwen3-emb revision=${CALLBUS_BACKBONE_REVISION:-} ;;
  *) name=${model//\//-} revision=${CALLBUS_BACKBONE_REVISION:-} ;;
esac
port=${VLLM_PORT:-8091}
args=(serve "$model" --runner pooling --max-model-len 8192 --no-enable-chunked-prefill
  --max-num-batched-tokens 8192 --host 127.0.0.1 --port "$port")
[[ -n "$revision" ]] && args+=(--revision "$revision")
if [[ "${CALLBUS_FP8:-1}" == 1 ]]; then
  served=${CALLBUS_EMB_MODEL:-$name-fp8}
  args+=(--quantization fp8)
else
  served=${CALLBUS_EMB_MODEL:-$name}
fi
args+=(--served-model-name "$served")
if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  args+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi
read -r -a extra <<<"${VLLM_EXTRA_ARGS:-}"
args+=(${extra[@]+"${extra[@]}"})
export CALLBUS_EMB_URL="http://127.0.0.1:$port/v1" CALLBUS_EMB_MODEL="$served"

if [[ -n "${CALLBUS_ENTRYPOINT_DRY_RUN:-}" ]]; then
  echo "vllm ${args[*]}"
  echo "CALLBUS_EMB_URL=$CALLBUS_EMB_URL CALLBUS_EMB_MODEL=$CALLBUS_EMB_MODEL callbus $*"
  exit 0
fi

vllm "${args[@]}" &
vllm_pid=$!
callbus_pid=
trap 'kill -TERM $vllm_pid $callbus_pid 2>/dev/null || true' TERM INT
echo "callbus-vllm: waiting for vLLM ($model as $served) on 127.0.0.1:$port" >&2
until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$port/health', timeout=2)" 2>/dev/null; do
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    wait "$vllm_pid" || exit $?
    exit 1
  fi
  sleep 2
done

callbus "$@" &
callbus_pid=$!
set +e
wait -n "$vllm_pid" "$callbus_pid"
status=$?
kill -TERM "$vllm_pid" "$callbus_pid" 2>/dev/null
wait
exit "$status"
