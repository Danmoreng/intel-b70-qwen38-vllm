#!/usr/bin/env python3
"""Refresh qualified maintenance measurements; preserve historical power data."""
import argparse
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    relative = str(root.relative_to(REPO))
    release = json.loads((REPO / 'config/production_image.json').read_text())
    old = json.loads((REPO / 'config/releases/exl3-v3/production_image.json').read_text())
    assessment, summary, promotion = [json.loads((root / name).read_text()) for name in
                                     ['assessment.json', 'serving-summary.json', 'promotion.json']]
    assert release['status'] == 'QUALIFIED_RELEASE'
    assert assessment['status'] == 'PASS_EXACT_IMAGE_RELEASE_GATES'
    assert promotion['status'] == 'PASS_STRICT_PRODUCTION_DEPLOYMENT_AND_RESTART'
    assert all(x['image_id'] == release['image_id'] for x in [assessment, summary, promotion])
    assert summary['policy_sha256'] == release['policy_sha256']
    policy = json.loads((REPO / 'config/production_policy.json').read_text())
    assert policy['runtime']['onednn'] == '3.13.4'
    spec = importlib.util.spec_from_file_location('stack_readme_tables', REPO / 'scripts/summarize-readme-benchmarks.py')
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    text = (REPO / 'README.md').read_text()
    power_start, power_end = '<!-- BEGIN POWER COMPARISON -->', '<!-- END POWER COMPARISON -->'
    historical_power = text.split(power_start, 1)[1].split(power_end, 1)[0]
    begin, end = '<!-- BEGIN CURRENT SERVING MEASUREMENTS -->', '<!-- END CURRENT SERVING MEASUREMENTS -->'
    assert text.count(begin) == text.count(end) == 1
    first, last = text.index(begin) + len(begin), text.index(end)
    text = text[:first] + '\n' + tables.readme_measurements(summary, root / 'serving-summary.json') + text[last:]
    assert old['image_id'] in text and old['image_tag'] in text
    text = text.replace(old['image_id'], release['image_id']).replace(old['image_tag'], release['image_tag'])
    text = text.replace('oneDNN 3.13.0; Intel Runtime', 'oneDNN 3.13.4; Intel Runtime')
    rollback_start = text.index('The immediate rollback is the ')
    rollback_end = text.index('The [EXL3 source snapshot/build instructions]', rollback_start)
    text = text[:rollback_start] + '''The immediate rollback is the [saved qualified EXL3 v3 release](config/releases/exl3-v3/README.md),
with its [strict launcher](scripts/run-server-exl3-v3-rollback.sh), policy,
source snapshot and compiler namespace. The older [v2](config/releases/exl3-v2/README.md)
and [v1](config/releases/exl3-v1/README.md) snapshots remain available.

''' + text[rollback_end:]
    start = text.index('See the [current v3 qualification and limitations]')
    finish = text.index('<!-- BEGIN CURRENT SERVING MEASUREMENTS -->', start)
    text = text[:start] + f'''This is a **oneDNN 3.13.4 maintenance release**, incorporating the upstream
SDPA initialization-order fix and a matching EXL3 rebuild. MTP3, original M04,
FP16, FP8 KV, 262,144 total tokens and 180 W remain unchanged. It is not a
demonstrated decode speedup. Eight independent paired core blocks give weighted
cold decode **+0.111%** (95% bootstrap interval −0.701% to +0.893%) and warm
decode **−0.752%** (−1.865% to +0.236%). Warm prose-C4 client throughput is
**−2.464%** (−5.363% to +0.132%); that adverse result remains visible and the
interval does not establish tight-margin noninferiority.

The full-matrix historical comparison flagged 4K/32K C1 decode losses and
4K C4 TTFT. Three balanced fresh-worker pairs retaining all original prompt
bytes, repeats and 1,024-output budgets did not reproduce a clear loss beyond
the pre-registered guardrails. Paired per-fixture native decode effects are
**+1.873%** at 4K (−2.160% to +4.211%) and **+1.404%** at 32K
(+0.581% to +2.945%). The 4K C4 client rate is **−0.923%**
(−1.413% to −0.152%), with finite-panel request E2E p95 **+0.882%**
(+0.122% to +1.544%). This smaller systematic C4 loss is an explicit
maintenance tradeoff. The historical adverse table and the alternate
within-arm median ratios remain published; these three paired starts are
not a formal universal noninferiority proof.
[Exact-fixture controls and counter-window limitations]({relative}/E01-frozen-trigger-reference-report.json).

See the [qualification and limitations]({relative}/assessment.json),
[full frozen-matrix comparison]({relative}/benchmark-comparison.json),
[reviewed final tradeoff]({relative}/performance-review.json), and
[overnight candidate decisions]({relative}/FINAL_DECISION.md).
The [public evidence manifest]({relative}/PUBLICATION_MANIFEST.json) maps
private receipt hashes to public counters and licensed runtime sources.

The exact short-array equality check failed and remains a failed receipt.
Original-tolerance independent references, actual-state replays, finite quality
panels and sampling/state checks support the separate numerical review;
they do not prove general task accuracy or identical generated text.
No BF16 model execution or FP16/BF16 serving bridge was introduced.

The [v3 qualification](benchmarks/results/vllm-031-upgrade/assessment.json)
and its recorded 64K acceptance/decode loss remain historical. The bounded
same-prefix investigation did not establish its upstream cause; no sampler
or state patch is claimed. The [v2 report](docs/EXL3_RELEASE_V2_REPORT.md),
[v1 report](docs/EXL3_RELEASE_REPORT.md), coding results and power comparisons
retain their original measurement scope. Coding tasks were not rerun here.

The serving tables below use fresh measurements of the exact deployed image.
[Runtime distribution and ABI pins](config/experiments/b70-stack-20261010/runtime_requirements.json)
record the unchanged Python stack and the rebuilt custom libraries. Existing
official-image metadata conflicts were not bypassed with forced installs.

''' + text[finish:]
    text = text.replace('This operating point is retained for vLLM 0.31; the power\ncampaigns remain the historical measurements below.',
                        'This operating point is retained for vLLM 0.31 / oneDNN 3.13.4.\nThe power campaigns were not repeated for this maintenance release; the\nhistorical measurements below remain on the older stack.')
    first = text.index('The new v3 run of the frozen short panel')
    last = text.index('## Rollback and benchmark reproduction', first)
    text = text[:first] + f'''The changed-stack short panel gives PPL **3.642167** versus stored v3
**3.642175**, with mean KL against the stored BF16 reference **0.033064**
versus **0.033353**. Short JS divergence slightly worsens; quality results are
mixed. The 32K suffix PPL is **1.183419** versus stored v3 **1.190357**.
These finite probes do not establish a general accuracy or coding improvement.
The BF16 reference was evaluated from stored arrays on the CPU.
[Current quality scope and metric receipts]({relative}/E01-numerical-review.json).

''' + text[last:]
    text = text.replace('The immediate rollback is the [saved EXL3 v2 release](config/releases/exl3-v2/production_image.json).\nUse `scripts/run-server-exl3-v2-rollback.sh`; it retains its own image, policy,\nsource snapshot and compiler namespace. The saved v1 and GPTQ releases also remain available.',
                        'The immediate rollback is the [saved qualified EXL3 v3 release](config/releases/exl3-v3/README.md).\nUse `scripts/run-server-exl3-v3-rollback.sh`; it retains its own image, policy,\nsource snapshot and compiler namespace. The saved v2, v1 and GPTQ releases also remain available.')
    current_power = text.split(power_start, 1)[1].split(power_end, 1)[0]
    assert [line for line in historical_power.splitlines() if line.startswith('|')] == [line for line in current_power.splitlines() if line.startswith('|')]
    (REPO / 'README.md').write_text(text)
    print('PASS_QUALIFIED_README_REFRESH_POWER_VALUES_UNCHANGED')


if __name__ == '__main__':
    main()
