#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
rollback_env="$repo_dir/config/releases/gptq-onednn-v2/.env"
[[ -f "$rollback_env" ]] || rollback_env="$repo_dir/config/releases/gptq-onednn-v2/.env.example"
if [[ -f "$rollback_env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$rollback_env"
  set +a
fi

require_value() {
  local name="$1" expected="$2" actual="${!1:-$2}"
  [[ "$actual" == "$expected" ]] || {
    echo "Production policy requires $name=$expected (got $actual)" >&2
    exit 2
  }
}
reject_value() {
  local name="$1"
  [[ -z "${!name:-}" || "${!name:-}" == 0 ]] || {
    echo "$name is a diagnostic/experimental override and cannot be used in production" >&2
    exit 2
  }
}

[[ "$#" == 0 ]] || { echo "Production launcher accepts no vLLM overrides" >&2; exit 2; }
require_value VLLM_IMAGE local/b70-qwen38-vllm:production-onednn-v2
require_value MODEL_ID mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
require_value MODEL_REVISION a47b0c6f0d756bc394c4cc629d5b0ded1acc7001
require_value SERVED_MODEL_NAME Qwen3.8-27B
require_value CONTEXT_SIZE 200704
require_value GPU_MEMORY_UTILIZATION 0.93
require_value MAX_NUM_BATCHED_TOKENS 6656
require_value MAX_NUM_SEQS 4
require_value SPECULATIVE_TOKENS 4
require_value SCHEDULER_WATERMARK 0.0
require_value PREFIX_CACHING 1
require_value B70_POWER_LIMIT_W 180
require_value B70_GPTQ_W4A8_PREFILL 1
require_value B70_GPTQ_W4A8_MIN_TOKENS 512
require_value B70_ONEDNN_PREFILL 1
require_value B70_ONEDNN_MIXED_ROUTE 1
require_value B70_ONEDNN_MIN_KV 16384
require_value B70_ONEDNN_MAX_KV 196608
require_value B70_DRAFT_VOCAB_ENABLED 0
require_value B70_DRAFT_LMHEAD_INT4 1
require_value B70_DRAFT_MTP_INT4 1
for flag in B70_ONEDNN_PROFILE B70_ONEDNN_SHORT_CHUNK_ONLY \
            B70_ONEDNN_FINAL_CHUNK_ONLY B70_ONEDNN_VALIDATE \
            B70_ONEDNN_VALIDATE_FP32 B70_ONEDNN_MIXED_TRACE \
            B70_ONEDNN_MIXED_VALIDATE B70_QUALIFICATION_ROUTE_OFF \
            B70_QUALIFICATION_CONTROL; do
  reject_value "$flag"
done

policy_sha="$(sha256sum "$repo_dir/config/releases/gptq-onednn-v2/production_policy.json" | cut -d' ' -f1)"
expected_policy_sha="$(cut -d' ' -f1 "$repo_dir/config/releases/gptq-onednn-v2/production_policy.sha256")"
[[ "$policy_sha" == "$expected_policy_sha" ]] || {
  echo "Production policy hash mismatch" >&2; exit 2;
}
image="${VLLM_IMAGE:-local/b70-qwen38-vllm:production-onednn-v2}"
image_id="$(docker image inspect "$image" --format '{{.Id}}')"
python3 - "$repo_dir/config/releases/gptq-onednn-v2/production_image.json" "$image" "$image_id" "$policy_sha" <<'PY'
import json, sys
from pathlib import Path
release = json.loads(Path(sys.argv[1]).read_text())
if (release['image_tag'], release['image_id'], release['policy_sha256']) != tuple(sys.argv[2:]):
    raise SystemExit('Production image differs from the frozen release identity')
PY
image_policy_sha="$(docker image inspect "$image" --format '{{index .Config.Labels "org.local.b70.policy.sha256"}}')"
[[ "$image_policy_sha" == "$policy_sha" ]] || {
  echo "Image policy hash mismatch: $image_policy_sha" >&2; exit 2;
}

render_node="${RENDER_NODE:-/dev/dri/renderD128}"
[[ -e "$render_node" ]] || { echo "Render node not found: $render_node" >&2; exit 2; }
power_caps=(/sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*/power1_cap)
[[ "${#power_caps[@]}" == 1 && -f "${power_caps[0]}" &&
   "$(cat "${power_caps[0]}")" == 180000000 ]] || {
  echo "B70 card power limit must be verified at 180 W" >&2; exit 2;
}
render_gid="$(stat -c '%g' "$render_node")"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/b70-qwen38-vllm-production/${policy_sha}/${image_id#sha256:}"
mkdir -p "$hf_home" "$cache_root/vllm" "$cache_root/triton"
echo "B70_PRODUCTION_LAUNCH policy=$policy_sha image=$image_id cache=$cache_root" >&2

exec docker run --rm --name "${VLLM_CONTAINER:-b70-qwen38-vllm}" \
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
  -e MODEL_ID=mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  -e MODEL_REVISION=a47b0c6f0d756bc394c4cc629d5b0ded1acc7001 \
  -e B70_GPTQ_W4A8_PREFILL=1 \
  -e B70_GPTQ_W4A8_MIN_TOKENS=512 \
  -e B70_ONEDNN_PREFILL=1 \
  -e B70_ONEDNN_MIXED_ROUTE=1 \
  -e B70_ONEDNN_MIN_KV=16384 \
  -e B70_ONEDNN_MAX_KV=196608 \
  -e B70_XPU_SINGLE_SEED_SAMPLER=0 \
  -e B70_MTP_BF16_DRAFT=1 \
  -e B70_DRAFT_LMHEAD_INT4=1 \
  -e B70_DRAFT_MTP_INT4=1 \
  -e B70_DRAFT_VOCAB_ENABLED=0 \
  -e B70_DRAFT_VOCAB_PATH= \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e PYTHONPATH=/opt/b70-runtime \
  "$image" "${MODEL_ID:-mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}" \
  --revision "${MODEL_REVISION:-a47b0c6f0d756bc394c4cc629d5b0ded1acc7001}" \
  --quantization gptq --dtype float16 \
  --max-model-len 200704 --gpu-memory-utilization 0.93 --kv-cache-dtype fp8 \
  --max-num-seqs 4 --max-num-batched-tokens 6656 \
  --scheduler-reserve-full-isl --watermark 0.0 \
  --enable-prefix-caching --mamba-cache-mode align \
  --served-model-name Qwen3.8-27B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --middleware request_defaults.ChatDefaults \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --override-generation-config '{"max_new_tokens":16384}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":4194304}'
