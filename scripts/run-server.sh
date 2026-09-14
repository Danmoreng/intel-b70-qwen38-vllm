#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_dir/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$repo_dir/.env"
  set +a
fi

image="${VLLM_IMAGE:-local/b70-qwen38-vllm:q128-196k-180w}"
model="${MODEL_ID:-mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
revision="${MODEL_REVISION:-a47b0c6f0d756bc394c4cc629d5b0ded1acc7001}"
served_name="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
render_node="${RENDER_NODE:-/dev/dri/renderD128}"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/b70-qwen38-vllm"
container="${VLLM_CONTAINER:-b70-qwen38-vllm}"

[[ -e "$render_node" ]] || { echo "Render node not found: $render_node" >&2; exit 2; }
[[ "${SPECULATIVE_TOKENS:-4}" =~ ^[1-9][0-9]*$ ]] || { echo "SPECULATIVE_TOKENS must be positive" >&2; exit 2; }
[[ "${MAX_NUM_BATCHED_TOKENS:-4096}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_NUM_BATCHED_TOKENS must be positive" >&2; exit 2; }

mkdir -p "$hf_home" "$cache_root/vllm" "$cache_root/triton"
render_gid="$(stat -c '%g' "$render_node")"

cache_args=(--enable-prefix-caching)
if [[ "${PREFIX_CACHING:-1}" != "1" ]]; then
  cache_args=(--no-enable-prefix-caching)
fi

speculative_config="{\"method\":\"mtp\",\"num_speculative_tokens\":${SPECULATIVE_TOKENS:-4}}"

exec docker run --rm --name "$container" \
  --device /dev/dri --group-add "$render_gid" \
  -v /dev/dri:/dev/dri:ro --shm-size 8g \
  -p "${VLLM_HOST:-127.0.0.1}:${PORT:-8081}:8000" \
  -v "$hf_home:/root/.cache/huggingface" \
  -v "$cache_root/vllm:/root/.cache/vllm" \
  -v "$cache_root/triton:/root/.triton/cache" \
  -v "$repo_dir/runtime:/opt/b70-runtime:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}" \
  -e B70_MTP_BF16_DRAFT=1 \
  -e B70_DRAFT_LMHEAD_INT4=1 \
  -e B70_DRAFT_MTP_INT4=1 \
  -e B70_DRAFT_VOCAB_ENABLED=0 \
  -e B70_DRAFT_VOCAB_PATH= \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e PYTHONPATH=/opt/b70-runtime \
  --entrypoint vllm "$image" serve "$model" \
  --revision "$revision" \
  --quantization gptq \
  --dtype float16 \
  --max-model-len "${CONTEXT_SIZE:-200704}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.93}" \
  --kv-cache-dtype fp8 \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  "${cache_args[@]}" \
  --mamba-cache-mode align \
  --served-model-name "$served_name" \
  --speculative-config "$speculative_config" \
  --middleware request_defaults.ChatDefaults \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --override-generation-config '{"max_new_tokens":16384}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":4194304}' \
  "$@"
