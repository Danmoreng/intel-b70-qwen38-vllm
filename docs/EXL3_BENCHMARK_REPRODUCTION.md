# Reproduce EXL3 serving measurements

Use exclusive GPU access, the qualified immutable image, a fixed 180 W limit,
frozen prompts/settings and fresh output directories. Model weights, large
original fixtures and raw events remain local. The fixture inventory preserves
248 file hashes. The measured matrix stays **20 scenarios / 70 waves / 124
requests at C1–C4**; admission remains 16 and the total-context limit 262,144.

## Refresh the current service measurements

First record the [current image/policy](../config/production_image.json),
[RAM overlay](../config/production_ram_kv.json), effective command, GPU power
and Git revision. Verify that the running container matches those identities.
Arrange exclusive GPU access and enough time for a bounded run, cleanup and
saving before the host's daily shutdown boundary. On the reference host,
start closing work at 22:45 Europe/Berlin, save and stop by 22:55, and run no
agent work beyond 23:00. Do not use a campaign to alter that schedule.

The original frozen fixture directory is **not included in Git**. Its manifest
records hashes, not recoverable prompt bytes. Obtain the original fixture
files before attempting an exact repeat; if generating new fixtures, label
that as a new workload and record its generation settings and hashes.

For a direct measurement on an idle current service, use the lower-level
runner. This does not stop or restore the service:

```bash
python3 scripts/current-profile-benchmark.py \
  --base http://127.0.0.1:8081 --container b70-qwen38-vllm \
  --expected-max-num-seqs 16 \
  --fixture-root /path/to/original/run-20260923-201101-w0.00 \
  --legacy-prefix-namespace \
  --output-root benchmark-results/serving-direct-new-run --execute
```

A matrix's 64K cold/warm row can be warmed by preceding scenarios. Restart the
service first, then repeat the command with `--only prefix-64k-cold-warm` and a
separate output directory to preserve an isolated observation. Do not publish
that direct run through the release exporter without its required provenance.

## Publishing a new current-profile run

The direct runner is not a release controller: it does not reserve the GPU for
you, stop competing work, enforce a whole-run deadline or restore a stopped
service. Bound the process externally, use fresh output directories, and check
health and a real reply after completion or interruption. Save image/policy,
overlay/command, power, fixture hashes, sampling, raw results and failed runs.

Publish throughput and semantic task scores separately. The current public
summary is linked from [BENCHMARKS.md](BENCHMARKS.md). When deliberately selecting
a new validated campaign, update the comparison source used by
`scripts/check-public-docs.py` and the dated methodology, then regenerate the
README table with `python3 scripts/check-public-docs.py --write`. This script
only formats existing evidence; it does not run or approve a benchmark.
The older `summarize-readme-benchmarks.py --update-readme` emits the previous
long-form table layout and should not overwrite the compact README section.

## Historical coding campaigns

`run-exl3-final-readme.py` is **historical v1 orchestration**: it unconditionally
runs both GPTQ and EXL3 Flappy campaigns. Do not invoke it to refresh current
EXL3 serving numbers. Its old paired measurements remain date-labeled in
[the coding history](EXL3_CODING_BENCHMARKS.md). `run-readme-benchmarks.py` also
includes QueueKit and is not the serving-only release workflow above.

The frozen v7 Flappy fixture preserves its original calibration instructions.
Its old `run-web-coding-campaign.py` refuses an EXL3 production service to avoid
mislabeling it as GPTQ. For an intentional future single-engine coding repeat,
use the standalone runner with the current service, keeping the six stages,
40-minute budget, 4K thinking budget and fresh-worker warmup unchanged. Such a
run needs its own engine identity and output; historical v1 results must not be
presented as measurements of a newer image.

## Historical v2 release controllers

The following workflow is retained for the old v2 release. It must not be used
as a current-profile exporter or as evidence for a new image.

### Serving-only release measurements

The bounded v2 controller measures the frozen final metadata child, checks its
parent filesystem identity and restores the qualified service on failure:

```bash
python3 scripts/run-exl3-review-serving.py \
  --release-dir config/experiments/exl3-review-release-v2 \
  --image-receipt benchmark-results/exl3-review-release-v2/image/image.json \
  --fixture-root /path/to/original/run-20260923-201101-w0.00 \
  --out benchmark-results/exl3-serving-new-run
python3 scripts/summarize-readme-benchmarks.py \
  benchmark-results/exl3-serving-new-run --serving-only \
  --output benchmark-results/exl3-serving-new-run/summary.json
```

The controller uses one fresh worker for the matrix and another for the
isolated 64K cold/warm prefix resend. Its saved image receipt is a local build
artifact; the published release manifest and assessment preserve its identities.
Do not synthesize a replacement receipt or weaken its checks. The controller
is specific to the frozen v2 review image, not a generic candidate selector.

### Additional gates

`scripts/run-exl3-review-qualification.py` runs operating checks, a small matched
optimized-v1/candidate screen, compact existing-reference quality checks and a
separate instrumented actual-worker cache diagnostic. It uses the same exclusive
GPU lock and explicit capacity 64. Failed checks restore the qualified service.
The existing BF16 references are reused. Only an initially triggered performance
case is repeated once; this is not a new kernel/MTP/vocabulary campaign.

`scripts/summarize-exl3-review-release.py` requires all gates and the final-image
serving campaign to pass before writing a promotion decision. The strict
promotion helper then checks the permanent service's generated replies,
streaming, API defaults and actual restart. A source archive alone is not a
release qualification.
