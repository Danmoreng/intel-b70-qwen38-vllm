#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_dir/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$repo_dir/.env"
  set +a
fi

run_dir="${1:-benchmark-results/$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "$run_dir" == /* || "$run_dir" == *".."* ]]; then
  echo "Output directory must be a relative path below the repository" >&2
  exit 2
fi

image="${VLLM_IMAGE:-local/b70-qwen38-vllm:q128-m04-196k-180w-runtime2635}"
model="${MODEL_ID:-mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
revision="${MODEL_REVISION:-a47b0c6f0d756bc394c4cc629d5b0ded1acc7001}"
served_name="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
api_root="${BENCHMARK_ROOT:-http://127.0.0.1:${PORT:-8081}}"
budget="${MAX_NUM_BATCHED_TOKENS:-6656}"

cd "$repo_dir"
mkdir -p "$run_dir"

docker run --rm \
  -v "$hf_home:/root/.cache/huggingface:ro" \
  -v "$repo_dir:/work" -w /work \
  --entrypoint python "$image" \
  scripts/generate-exact-prompts.py \
  --model "$model" --revision "$revision" \
  --targets 512,8192,32768,65536,131072 \
  --per-target 6 --output "/work/$run_dir/prompts.json"

python3 scripts/context-benchmark.py \
  --mode context --prompts "$run_dir/prompts.json" \
  --outdir "$run_dir/p512-g128" \
  --model "$served_name" --budget "$budget" --reps 5 \
  --target 512 --output 128 --root "$api_root" \
  --full-output-warmup --ignore-eos

for target in 8192 32768 65536 131072; do
  python3 scripts/context-benchmark.py \
    --mode context --prompts "$run_dir/prompts.json" \
    --outdir "$run_dir/p${target}-g512" \
    --model "$served_name" --budget "$budget" --reps 5 \
    --target "$target" --output 512 --root "$api_root" \
    --full-output-warmup --ignore-eos
done

echo "Benchmark complete: $run_dir"
