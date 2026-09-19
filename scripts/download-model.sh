#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_dir/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$repo_dir/.env"
  set +a
fi

image="${VLLM_IMAGE:-local/b70-qwen38-vllm:q128-m04-196k-180w}"
model="${MODEL_ID:-mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
revision="${MODEL_REVISION:-a47b0c6f0d756bc394c4cc629d5b0ded1acc7001}"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$hf_home"

docker run --rm \
  -v "$hf_home:/root/.cache/huggingface" \
  --entrypoint hf "$image" \
  download "$model" --revision "$revision"

