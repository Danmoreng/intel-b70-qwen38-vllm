#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_dir/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$repo_dir/.env"
  set +a
fi

default_image="local/b70-qwen38-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4"
image="${VLLM_IMAGE:-$default_image}"
model="${MODEL_ID:-mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
revision="${MODEL_REVISION:-a47b0c6f0d756bc394c4cc629d5b0ded1acc7001}"
served_name="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
render_node="${RENDER_NODE:-/dev/dri/renderD128}"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
container="${VLLM_CONTAINER:-b70-qwen38-vllm}"

[[ -e "$render_node" ]] || { echo "Render node not found: $render_node" >&2; exit 2; }
speculative_tokens="${SPECULATIVE_TOKENS:-4}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-6656}"
max_num_seqs="${MAX_NUM_SEQS:-4}"
scheduler_watermark="${SCHEDULER_WATERMARK:-0.0}"
w4a8_prefill="${B70_GPTQ_W4A8_PREFILL:-0}"
onednn_prefill="${B70_ONEDNN_PREFILL:-0}"
onednn_profile="${B70_ONEDNN_PROFILE:-reference}"
case "$onednn_profile" in
  reference) onednn_short_default=1; onednn_max_default=131072 ;;
  performance) onednn_short_default=0; onednn_max_default=200704 ;;
  *) echo "B70_ONEDNN_PROFILE must be reference or performance" >&2; exit 2 ;;
esac
onednn_short="${B70_ONEDNN_SHORT_CHUNK_ONLY:-$onednn_short_default}"
onednn_min_kv="${B70_ONEDNN_MIN_KV:-16384}"
onednn_max_kv="${B70_ONEDNN_MAX_KV:-$onednn_max_default}"
[[ "$w4a8_prefill" =~ ^[01]$ && "$onednn_prefill" =~ ^[01]$ &&
   "$onednn_short" =~ ^[01]$ && "$onednn_min_kv" =~ ^[1-9][0-9]*$ &&
   "$onednn_max_kv" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid W4A8/oneDNN cache variant flags" >&2; exit 2;
}
cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/b70-qwen38-vllm"
if [[ "$image" != "$default_image" || "$w4a8_prefill" == 1 || "$onednn_prefill" == 1 ]]; then
  # AOT artifacts can otherwise be reused across different linear/attention
  # selections. Keep the unchanged production cache path, isolate experiments.
  image_id="$(docker image inspect "$image" --format '{{.Id}}')"
  cache_variant="w4a8${w4a8_prefill}-onednn${onednn_prefill}-${onednn_profile}-short${onednn_short}-min${onednn_min_kv}-max${onednn_max_kv}"
  cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/b70-qwen38-vllm-experiment/${image_id#sha256:}/$cache_variant"
fi
[[ "$speculative_tokens" =~ ^[1-9][0-9]*$ ]] || { echo "SPECULATIVE_TOKENS must be positive" >&2; exit 2; }
[[ "$max_num_batched_tokens" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_NUM_BATCHED_TOKENS must be positive" >&2; exit 2; }
[[ "$max_num_seqs" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_NUM_SEQS must be positive" >&2; exit 2; }
[[ "$scheduler_watermark" =~ ^0(\.[0-9]+)?$ ]] || { echo "SCHEDULER_WATERMARK must be in [0.0, 1.0)" >&2; exit 2; }

mkdir -p "$hf_home" "$cache_root/vllm" "$cache_root/triton"
render_gid="$(stat -c '%g' "$render_node")"

cache_args=(--enable-prefix-caching)
if [[ "${PREFIX_CACHING:-1}" != "1" ]]; then
  cache_args=(--no-enable-prefix-caching)
fi

speculative_config="{\"method\":\"mtp\",\"num_speculative_tokens\":$speculative_tokens}"

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
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e B70_GPTQ_W4A8_PREFILL="$w4a8_prefill" \
  -e B70_ONEDNN_PREFILL="$onednn_prefill" \
  -e B70_ONEDNN_PROFILE="$onednn_profile" \
  -e B70_ONEDNN_VALIDATE="${B70_ONEDNN_VALIDATE:-0}" \
  -e B70_ONEDNN_VALIDATE_FP32="${B70_ONEDNN_VALIDATE_FP32:-0}" \
  -e B70_ONEDNN_MIN_KV="$onednn_min_kv" \
  -e B70_ONEDNN_MAX_KV="$onednn_max_kv" \
  -e B70_ONEDNN_SHORT_CHUNK_ONLY="$onednn_short" \
  -e B70_ONEDNN_FINAL_CHUNK_ONLY="${B70_ONEDNN_FINAL_CHUNK_ONLY:-0}" \
  -e B70_XPU_SINGLE_SEED_SAMPLER=0 \
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
  --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens "$max_num_batched_tokens" \
  --scheduler-reserve-full-isl \
  --watermark "$scheduler_watermark" \
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
