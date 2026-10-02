#!/usr/bin/env python3
"""Refresh operations README/report from qualified v2 measurements only."""
import argparse
import importlib.util
import json
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);a=p.parse_args();root=a.results.resolve()
    assessment=json.loads((root/'assessment.json').read_text());summary=json.loads((root/'serving-summary.json').read_text())
    release=json.loads((REPO/'config/production_image.json').read_text());promotion=json.loads((root/'promotion.json').read_text())
    assert assessment['status']=='PASS_EXACT_IMAGE_RELEASE_GATES'
    assert promotion['status']=='PASS_STRICT_PRODUCTION_DEPLOYMENT_AND_RESTART' and release['status']=='QUALIFIED_RELEASE'
    assert assessment['image_id']==summary['image_id']==release['image_id']==promotion['image_id']
    assert assessment['policy_sha256']==summary['policy_sha256']==release['policy_sha256']
    module=importlib.util.spec_from_file_location('serving_tables',REPO/'scripts/summarize-readme-benchmarks.py')
    tables=importlib.util.module_from_spec(module);module.loader.exec_module(tables)
    text=(REPO/'README.md').read_text()
    old=json.loads((REPO/'config/releases/exl3-v1/production_image.json').read_text())
    text=text.replace(old['image_id'],release['image_id']).replace(old['image_tag'],release['image_tag'])
    beginning='<!-- BEGIN CURRENT SERVING MEASUREMENTS -->';ending='<!-- END CURRENT SERVING MEASUREMENTS -->'
    assert text.count(beginning)==text.count(ending)==1
    start=text.index(beginning)+len(beginning);end=text.index(ending)
    text=text[:start]+'\n'+tables.readme_measurements(summary,root/'serving-summary.json')+text[end:]
    row='| Verification | Rebuilt M04 for supported uniform q2–5 / C1–C4; native fallback elsewhere; no duplicate KV updates |'
    assert row in text
    text=text.replace(row,'| Verification | M04 for supported uniform q2–5 / C1–C4; direct output copy at C4; native fallback elsewhere; no duplicate KV updates |\n| oneDNN partition cache | 64 exact-shape entries per inference thread/queue; completion-aware LRU eviction; oneDNN internal caches are a separate scope |')
    preemptions=int(assessment['operations']['cases']['c16-long']['native']['preemptions'])
    text=re.sub(r'  requests share the same cache pool\. C16 pressure tests completed with four\n  preemptions affecting two requests; this allows waiting/recomputation and\n',
        f'  requests share the same cache pool. All 16 pressure-test requests completed,\n  with {preemptions} preemptions; this allows waiting/recomputation and\n',text)
    text=text.replace('  rounding. The short quality panel does not exercise the ≥4096-token oneDNN\n  prefill route or prove identical graph-based MTP verification.','  rounding. Fresh candidate reference checks and separate long-prefill/C4\n  graph numerical probes pass; finite probes do not guarantee identical text.')
    text=text.replace('- QueueKit passes **7/8 checks and 1/2 tasks**. The same failed case occurs in\n  the native-attention control; the actual task failure remains documented.',
        '- Historical EXL3 v1 QueueKit passed **7/8 checks and 1/2 tasks**, with the\n  same failure in its native-attention control. Coding tasks were not rerun\n  for this release; the failed task remains documented.')
    text=text.replace('See the [qualification and limitations](docs/EXL3_RELEASE_REPORT.md) and the\n[bounded Pro-review follow-up](docs/EXL3_PRO_REVIEW_FOLLOWUP.md). Review candidate\nmeasurements are kept separate from the immutable v1 production measurements\nbelow.',
        'See the [current v2 qualification and limitations](docs/EXL3_RELEASE_V2_REPORT.md).\nThe [v1 report](docs/EXL3_RELEASE_REPORT.md), coding measurements and\n[bounded review experiments](docs/EXL3_PRO_REVIEW_FOLLOWUP.md) remain historical.\nThe serving measurements below are fresh measurements of the deployed v2 image.')
    text=text.replace('The [EXL3 source snapshot/build instructions](engine/exl3xpu/README.md) contain',
        'The [saved EXL3 v1 rollback](config/releases/exl3-v1/README.md) has its own\nstrict launcher, policy, source snapshot and compiler namespace. Its immutable\ntag is retained alongside the current image.\n\nThe [EXL3 source snapshot/build instructions](engine/exl3xpu/README.md) contain')
    (REPO/'README.md').write_text(text)
    relative=str(root.relative_to(REPO));quality=assessment['quality_smoke'];long=quality['long'];matched=assessment['matched_comparison']
    body=f'''# EXL3 v2 release qualification — 2026-10-02

The bounded Pro-review changes are qualified and deployed. This release adds
the completion-aware exact-shape partition cache, C4-only direct M04 output
copy and one guarded route decision. Quantization, MTP3, draft vocabulary,
attention formulas, split table and toolchain remain frozen.

## Exact release identity

- Production image: `{release['image_id']}`, alias `{release['image_tag']}`.
- Reviewed parent: `{release['runtime_image_id']}`; filesystem layers identical.
- Explicit partition-cache capacity: 64 per thread/queue. The metadata child
  adds the policy identity and explicit cache environment, without a native rebuild.
- Policy: `{release['policy_sha256']}`.
- EXL3 source: `{release['source_commit']}`; native library
  `{release['runtime_artifacts_sha256']['/opt/exl3xpu/exl3xpu/_C.so']}`.
- M04 library remains
  `{release['runtime_artifacts_sha256']['/opt/exl3xpu/m04/m04.so']}`.
- vLLM 0.30.0 / Torch 2.13.0+xpu / oneAPI 2026.1.1 / oneDNN 3.13.0, 180 W.

The [release assessment](../{relative}/assessment.json),
[decision](../{relative}/decision.json) and
[actual service startup/restart receipt](../{relative}/promotion.json)
preserve identities, configuration, source hashes, tests and limitations.

The first cache-diagnostic worker failed before model startup because a file
mount targeted a read-only parent directory. Its failed campaign receipt is
retained. Only that diagnostic phase was rerun with a separate read-only
diagnostic directory; already passed serving, operating and quality gates
were preserved under their original identities. No image bytes changed.

## Fresh serving measurements

The existing 20-scenario / 70-wave / 124-request matrix ran once on the final
deployable image, with identical frozen prompt bytes and request settings.
It completed in {summary['source_review']['wall_s']/60:.2f} minutes with zero
preemptions or excess recomputed prefill tokens. A separate fresh-worker 64K
prefix resend preserves the cold/warm observation. Adaptive Flappy, GPTQ,
MTP/vocabulary studies and original BF16 inference were not rerun.
See the [new serving summary](../{relative}/serving-summary.json).

The matched v1/candidate screen uses both intended optimized profiles, common
frozen performance windows, fixed temperature/seed and separate warmup.
Short C1, 103K C1, short C4, 32K C4 and mixed C4 are measured. Review triggers
are >5% slower decode/TTFT or >10% higher mixed client p95 burst gap; only an
initially triggered case is repeated once, in reversed image order. These are
bounded decision triggers, not an SLA or a statistical distribution.

| Case / pass | Decode change | TTFT change | Mixed p95 gap change | Trigger |
|---|---:|---:|---:|---|
'''
    for pass_name in ('initial','repeat'):
        for r in matched[pass_name]:
            def pct(key):return f"{r[key]:+.2f}%" if key in r else '—'
            body+=f"| {r['label']} / {pass_name} | {pct('decode_change_pct')} | {pct('ttft_change_pct')} | {pct('p95_stream_gap_change_pct')} | {r['trigger']} |\n"
    body+=f'''
No unresolved repeated trigger remains. Candidate microcopy measurements
around +0.8% at short C4 are not published as serving throughput gains.
Equivalent serving performance plus bounded resource retention can justify
this release; no universal speedup is claimed.

## Operating and numerical gates

- Fresh load: 409/409 modules, including 8/8 MTP; all 13 active source guards
  and their complete runtime-source capture pass.
- API: generated/streaming text, automatic tools, reasoning, image/video
  count limits and capped images pass. Permanent-service checks include real
  default-reasoning requests, exact image/policy identity and a service restart.
- Exact context boundary: 261,120 input + 1,024 output, no preemption.
- Moderate C4, long-image context, prefix extension, scheduler-confirmed
  abort/recovery, independent maximum-area images and worker restart pass.
- C16 pressure: all 16 independent 8K + 256-output requests complete;
  {preemptions} preemptions are permitted and recorded. This does not promise
  sixteen simultaneous maximum contexts or preemption-free C16.
- Fresh short-reference smoke: {quality['short']['ppl_positions']:,} scored
  positions, PPL {quality['short']['perplexity']:.6f}, mean BF16-relative KL
  {quality['short']['kl_mean']:.6f}; all 32 aggregate arrays bitidentical to
  v1. Original reference data is reused. This short panel alone does not
  exercise long oneDNN prefill or prove MTP graph correctness.
- Fresh existing 32K-prefix suffix smoke: {long['suffix_positions']} scored
  positions / {long['full_vocabulary_positions']} full distributions, PPL
  {long['perplexity']:.6f} versus BF16 {long['original_perplexity']:.6f},
  mean KL {long['kl_mean']:.6f}. It is a compact engineered window, not a new
  broad model-quality campaign.
- Separate fresh candidate matched-state probes exercise actual long prefill,
  mixed serving and true C4 verification graphs. Every local comparison passes
  the original rtol 0.01 / atol 0.003. Histories/states are restored within
  each finite probe; no historical bit-exact-text or universal determinism claim.

## Actual serving-worker cache observation

A separate instrumented worker performs mixed decode/incoming 49K prefill and
70 varied exact prompt lengths plus repeats. Counters are read on the actual
inference thread/current queue after its native attention enqueue. This is
not a separate-process cache query. Each observed cache stays at/below 64,
with hits, misses and eviction; pressure waits, compilation microseconds,
RSS and XPU allocated/reserved memory are preserved in the assessment.
These instrumented times are excluded from the throughput comparison.

The cap applies to each application thread/queue context. It does not globally
bound arbitrary thread/queue counts, oneDNN internal caches or every allocator,
and finite measurements do not guarantee permanent freedom from OOM.
Client p95 SSE burst gaps are delivery observations, not GPU token-step times.

## Historical coding results and rollback

The v1 Flappy/QueueKit runs remain [separately dated historical evidence](EXL3_CODING_BENCHMARKS.md).
QueueKit's 7/8 checks and 1/2 tasks, including the same native-control failure,
remain visible. No coding task was rerun or relabeled for v2.

The immediately usable [EXL3 v1 rollback](../config/releases/exl3-v1/README.md)
retains image `{old['image_id']}`, its immutable tag, policy, original release
receipts, source and runtime snapshots and compiler namespace. The matched
v1 worker loads and serves real requests during this release qualification.
The strict rollback preflight is separately checked. A failed deployment
automatically restores v1 configuration and the local environment.
No sudo, driver or system-service installation was needed.
'''
    (REPO/'docs/EXL3_RELEASE_V2_REPORT.md').write_text(body)
    (root/'README.md').write_text('# Qualified EXL3 v2 release evidence\n\nSee the [release report](../../../docs/EXL3_RELEASE_V2_REPORT.md), serving-summary.json, assessment.json, decision.json and promotion.json. Throughput is measured on the exact final image; numerical and cache diagnostics are separately scoped. Historical coding runs are not relabeled.\n')


if __name__=='__main__':main()
