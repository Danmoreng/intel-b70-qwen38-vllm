# M13: seeded Wikipedia summarization with MTP

This experiment compares the exact production vLLM 0.29 image with the M12
vLLM 0.30 candidate under a German Wikipedia summarization workload. Both
images retain the production Intel runtime, cookbook and local patches,
Q128/M04 binaries, model revision, MTP4, FP8 KV, 180 W cap and serving flags.

Four German Wikipedia revisions are pinned in `prepare.py`: Marie Curie,
Apollo 11, Photosynthese and Berlin. It fetches parsed revision HTML through
the [MediaWiki Parse API](https://www.mediawiki.org/wiki/API:Parsing_wikitext),
extracts paragraphs, and freezes roughly 4K-token prompts in the ignored
local benchmark directory. The tracked result contains article revision URLs
and SHA-256 hashes, not copied Wikipedia text.

The [vLLM Chat API](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/serving/online_serving/openai_compatible_server.md)
receives `temperature=0.7`, `top_p=0.9`, `max_tokens=512`, a fixed seed per
request, and `chat_template_kwargs={"enable_thinking":false}` to generate the
requested summary rather than spending the output budget on reasoning. One
C4 warmup precedes four C1 waves (one per article) and two waves each for
C2–C4. Prefix cache is reset before every wave. B/A/A/B order uses separate
compiler caches by version and restores production on success or failure.

From the repository root:

```bash
python3 benchmarks/experiments/m13-wikipedia-mtp/prepare.py \
  --output benchmark-results/wiki-mtp-prompts/prompts.json
python3 benchmarks/experiments/m13-wikipedia-mtp/run_ab.py \
  --run-dir benchmark-results/wiki-mtp-ab \
  --prompts benchmark-results/wiki-mtp-prompts/prompts.json \
  --execute
python3 benchmarks/experiments/m13-wikipedia-mtp/summarize.py \
  --run-dir benchmark-results/wiki-mtp-ab \
  --output benchmarks/runs/wiki-mtp/summary.json
```

The completed run is at
[`../../runs/2026-09-23-wikipedia-mtp/README.md`](../../runs/2026-09-23-wikipedia-mtp/README.md).
