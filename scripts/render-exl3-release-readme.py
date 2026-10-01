#!/usr/bin/env python3
"""Render the qualified EXL3 README from independently validated result exports."""
import argparse
import importlib.util
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--flappy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    summary=json.loads(a.summary.read_text());flappy=json.loads(a.flappy.read_text())
    release=json.loads((REPO/'config/production_image.json').read_text())
    policy=json.loads((REPO/'config/production_policy.json').read_text())
    assert release['status']=='QUALIFIED_RELEASE' and summary['image_id']==release['image_id']
    assert summary['policy_sha256']==release['policy_sha256']
    assert flappy['engines']['exl3']['identity']['image_id']==release['image_id']
    spec=importlib.util.spec_from_file_location('release_readme_tables',REPO/'scripts/summarize-readme-benchmarks.py')
    tables=importlib.util.module_from_spec(spec);spec.loader.exec_module(tables)
    body=tables.readme_measurements(summary,a.summary.resolve())
    body=body.replace('## Repeatable coding-agent benchmark','## Coding-agent benchmarks\n\n### Short Python fixture')
    historical=json.loads((REPO/'benchmarks/runs/2026-09-30-production/summary.json').read_text())
    old={x['name']:x for x in historical['source_review']['scenario_results']}
    new={x['name']:x for x in summary['source_review']['scenario_results']}
    text=f'''# Intel Arc Pro B70: Qwen3.8-27B with vLLM

The current production profile serves **EXL3 4.00 bpw on one 32 GB Intel Arc Pro B70 at 180 W**.
It preserves the OpenAI-compatible API and uses the pinned vLLM 0.30/Torch 2.13
port, native EXL3 row dispatch, guarded exact-K oneDNN prefill and rebuilt M04
shared-KV verification. The previous GPTQ image remains an independently pinned rollback.

## Current production configuration

| Setting | Applied value |
|---|---|
| Model / revision | `turboderp/Qwen3.8-27B-exl3` / `{policy['model']['revision']}` |
| Served name | `Qwen3.8-27B` |
| Weights / activations / target head | EXL3 4.00-bpw checkpoint / FP16 / 6-bpw full 248,320-row head |
| Linear dispatch | Native EXL3 SmallM through 128 rows; INT8 large-matrix prefill; unchanged row policy |
| Prefill attention | Guarded oneDNN: query rows ≥64, exact active KV 4,096–262,144, query bucket 256, exact-K bucket 1; eager only |
| Verification | Rebuilt M04 for supported uniform q2–5 / C1–C4; native fallback elsewhere; no duplicate KV updates |
| MTP | 3 draft tokens; 65,536-row draft vocabulary; full 248,320-row target vocabulary |
| Context | 262,144 total input+output tokens |
| Admission / batch | 16 sequences; 4,096 max batched tokens; full-ISL, watermark 0.0 |
| Graphs / cache | FULL_DECODE_ONLY, capture sizes 1/2/4/8/12/16/24/32/40/48/56/64; FP8 KV, prefix reuse, Mamba alignment |
| Memory / power | 0.965 GPU fraction / 180 W card cap |
| Runtime | vLLM 0.30.0, Torch 2.13.0+xpu, oneAPI 2026.1.1 / SYCL 9, oneDNN 3.13.0; Intel Runtime 26.35.39758.10, IGC 2.41.5 |
| Tools / reasoning / media | qwen3_xml / qwen3; 32 images or 4 videos within the total context limit; image cap 4,194,304 pixels |

The [policy](config/production_policy.json) is `{release['policy_sha256']}`.
The [release image](config/production_image.json) is `{release['image_id']}`
(`{release['image_tag']}`). The final benchmark used this immutable image with
its candidate alias; promotion adds an alias and does not rebuild the payload.
The launcher verifies image/policy labels, native/M04/oneDNN/profile artifacts,
checkpoint file hashes and middleware before loading the model. Compiled caches
are isolated by policy and image. The complete EXL3 source snapshot, upstream
patch and build pins are published in [engine/exl3xpu](engine/exl3xpu/README.md).

C4 moderate-context qualification is clean. C16 permits genuine pool pressure:
all 16 independent 8K + 256-token requests completed, with four preemptions/two
affected requests and 16,000 extra submitted prefill tokens. Aligned Mamba and
speculative states share the block pool; this is not 16 simultaneous 262K contexts.
The exact 261,120 + 1,024-token boundary passes without preemption.32 different
4.2 MP images (131,164 input tokens), video limits, long-image context,
prefix extension, scheduler-confirmed abort/recovery and independent restart
pass. See the [release report](docs/EXL3_RELEASE_REPORT.md).

## Quality and optimization evidence

The same frozen short panel retains the observed EXL3 quality advantage:
original BF16 PPL 3.60453, historical supplied GPTQ PPL 3.80193/KL 0.086369,
current EXL3 precision path PPL 3.64672/KL 0.032481. These are finite-panel
checkpoint/path measurements, not a general coding ranking or an isolated
quantization-only comparison. Matched native/optimized EXL3 has 272 bit-identical
short-panel arrays. Four engineered 32K/100K/180K/262K prefixes yield suffix
PPL 1.20858 versus BF16 1.21075, KL 0.001178 and 32/32 sampled top1 agreement.
Generated histories can differ; no bit-exact-text guarantee is claimed.
[Quality scope and raw metric receipts](benchmarks/results/exl3-migration/optimized-quality-v1/README.md).

All 409 weight tensors including 8 MTP tensors reconstruct bit-exactly. Seven
linear classes, the full target head, 18 row counts, tail poisoning, large-first
compilation and supported graphs pass unchanged numerical tolerances. Large
INT8-prefill capture is unsupported and outside the frozen decode-only profile.
The brief final kernel profile did not justify a speculative rewrite.

Mixed ABBA improves incoming 49K TTFT ~36→27 s and overlapping SSE gap p95 ~3.0→2.05 s.
Repeated 103K C1 whole-attention ABBA improves decode 43.98→49.24 tok/s cold
and 43.91→49.24 warm (~12%), with matched cache residency. Histories differ,
so this is not an isolated kernel gain. The separate M04-only proof has its own
~14% / identical 512-token scope. [Measured optimization evidence](benchmarks/results/exl3-migration/optimized-performance-v1/README.md).

'''+body
    text+='''### Long WebGL2 coding task

The corrected [Flappy Bird v7 assignment](benchmarks/web-coding-fixture/v7/README.md)
uses six fixed stages, deterministic physics, procedural WebGL2 graphics,
controls, responsive UI, settings and highscores, without a level editor or
replay system. Both engines use the same task/harness, seeds, sampling,
retained reasoning, 4,096-token thinking budget and 40-minute task budget.
Unmodified final outputs are independently graded against the same 54 cases.

| Result | GPTQ production v2 | Current EXL3 v1 |
|---|---:|---:|
'''
    g,e=[flappy['engines'][k] for k in ['gptq','exl3']]
    def wall(x):return f"{int(x['wall_s']//60)}min {x['wall_s']%60:.0f}s; {x['session_status']}"
    rows=[('Wall time / task outcome',wall(g),wall(e)),('Requests / maximum input context',f"{g['requests']} / {g['overall']['context_max']:,}",f"{e['requests']} / {e['overall']['context_max']:,}"),
          ('Frozen functional checks',f"{g['quality']['passed']}/{g['quality']['total']}",f"{e['quality']['passed']}/{e['quality']['total']}"),
          ('Native prefill / decode tok/s',f"{g['overall']['prefill_tps']:.1f} / {g['overall']['decode_tps']:.1f}",f"{e['overall']['prefill_tps']:.1f} / {e['overall']['decode_tps']:.1f}")]
    for label,x,y in rows:text+=f'| {label} | {x} | {y} |\n'
    public_flappy=str(a.flappy.parent.resolve().relative_to(REPO))
    text+=f'''
[All measured 10K context bands and request accounting]({public_flappy}/README.md)
preserve empty bands as unmeasured. Prefill counts new KV tokens; decode counts
post-first generated tokens including reasoning. Agent histories differ, so
overall rates and task duration do not isolate engine or quantization effects.
One seed/pair does not establish a general model-quality ranking. The older
[v6 result](benchmarks/runs/2026-10-01-flappybird/README.md) remains historical;
it is not substituted for the corrected current run.

![Rates over the growing coding context]({public_flappy}/context-rates.png)

## Comparison with the previous GPTQ profile

GPTQ numbers are the published 2026-09-30 full run; EXL3 numbers are the new
complete run using identical frozen request payloads. The GPTQ full matrix was
not repeated. Its corrected v7 coding task above is a new paired measurement.
Both profiles run at 180 W; MTP depth and quantized checkpoints differ. Remaining
decode differences are shown explicitly alongside the quality/context gain.

| Frozen input budget | GPTQ prefill | EXL3 prefill | GPTQ decode | EXL3 decode | Decode change |
|---:|---:|---:|---:|---:|---:|
'''
    for name in ['phase-512-c1','phase-2k-c1','phase-4k-c1','phase-8k-c1','phase-16k-c1','phase-32k-c1','phase-64k-c1','phase-128k-c1','full-context-199680']:
        x,y=old[name],new[name]
        text+=f"| {y['actual_prompt_tokens_min']:,}–{y['actual_prompt_tokens_max']:,} | {x['prefill_tps_median']:.1f} | {y['prefill_tps_median']:.1f} | {x['decode_tps_median']:.1f} | {y['decode_tps_median']:.1f} | {100*(y['decode_tps_median']/x['decode_tps_median']-1):+.1f}% |\n"
    text+='''
## Install, serve and reproduce

Build instructions and immutable upstream/native/header pins are in
[engine/exl3xpu](engine/exl3xpu/README.md). This local release pins an already
qualified image; a different rebuilt image needs its own qualification and
manifest update. Model weights and large local fixtures are not redistributed.

```bash
cp .env.example .env
./scripts/download-model.sh
./scripts/set-power-limit.py
./scripts/install-user-service.sh
curl -fsS http://127.0.0.1:8081/v1/models
```

The API is `http://127.0.0.1:8081/v1`, model `Qwen3.8-27B`.
`scripts/run-server.sh` selects the frozen production profile. The independent
[GPTQ rollback launcher/config](config/releases/gptq-onednn-v2/README.md)
does not share EXL3 compiled caches. Stop the service before swapping profiles.

To repeat the full source matrix and QueueKit on the current permanent service:

```bash
python3 scripts/run-readme-benchmarks.py \
  --fixture-root /path/to/original/run-20260923-201101-w0.00 \
  --output-root benchmark-results/readme-new-run
```

The admission check reads the frozen policy (16 for this profile); the measured
matrix stays 20 scenarios / 70 waves / 124 requests at C1–C4. Use a fresh idle worker
for isolated 64K cold/warm resends with `current-profile-benchmark.py --container b70-qwen38-vllm
--expected-max-num-seqs 16 --only prefix-64k-cold-warm`. The full paired release
controller and validated exports are in `scripts/run-exl3-final-readme.py`,
`summarize-readme-benchmarks.py` and `summarize-web-coding-benchmark.py`.
The large original fixtures/raw events remain local with 248 frozen file hashes.

## Sources and acknowledgements

- [Qwen model](https://huggingface.co/Qwen/Qwen3.8-27B)
- [EXL3 checkpoint](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
- [0xSero EXL3 XPU source](https://github.com/0xSero/exl3xpu)
- [vLLM](https://github.com/vllm-project/vllm)
- [Intel B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)

See [NOTICE.md](NOTICE.md) and the preserved upstream license for attribution.
'''
    a.output.write_text(text)


if __name__=='__main__':main()
