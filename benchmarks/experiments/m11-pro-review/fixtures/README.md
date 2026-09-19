# Reproduce the local request fixtures

`fixtures.json` is generated and ignored (about 9 MB of repeated prompt text).
`manifest.json` retains its digest and the individual request hashes. The
generator is `../make-fixtures.py`; the tokenizer is pinned to model revision
`a47b0c6f0d756bc394c4cc629d5b0ded1acc7001`.

From the repository root, with the documented production image and model cache:

```bash
repo="$PWD"
experiment="$repo/benchmarks/experiments/m11-pro-review"
model_repo="$HOME/.cache/huggingface/hub/models--mikeinnyc--Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
docker run --rm --network=none --entrypoint python \
  -v "$repo/scripts:/scripts:ro" \
  -v "$experiment:/src:ro" -v "$experiment/fixtures:/out" \
  -v "$model_repo/snapshots/a47b0c6f0d756bc394c4cc629d5b0ded1acc7001:/model:ro" \
  -v "$model_repo/blobs:/blobs:ro" \
  local/qwen38-b70-vllm:runtime-26.35-igc-2.41.5-20260919 \
  /src/make-fixtures.py
sha256sum "$experiment/fixtures/fixtures.json"
```

Expected SHA-256:
`12feda4f441e48e6aff8c5cb5ab3306b5020a66952e309a0f52bc79c728aa1d7`.
No GPU is needed for fixture generation. The serving harness uses repetition
zero for warmup, then repetitions one through five for paired measurements.
