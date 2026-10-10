"""Record an explicit hashed numerical review; preserve the failed exact gate."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
p=argparse.ArgumentParser();p.add_argument('--reviewed-measurement-sha256',required=True);a=p.parse_args()
root=REPO/'benchmark-results/b70-stack-20261010';q=root/'E01-quality'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(q/'changed-stack-quality.json')==a.reviewed_measurement_sha256
measurement=json.loads((q/'changed-stack-quality.json').read_text());capture=json.loads((q/'campaign.json').read_text());replay=json.loads((q/'replay-campaign.json').read_text());linear=json.loads((q/'linear/result.json').read_text());gates=json.loads((root/'E01-gpu-gates/campaign.json').read_text());edge=json.loads((q/'sdpa-edge-values.json').read_text());d64=json.loads((REPO/'benchmarks/results/b70-stack-20261010/D64-same-prefix-review.json').read_text())
image='sha256:b772d37937d188bc8492e2e796eafc46668290e68d356f648257e99b06967048'
assert capture['image_id']==replay['image_id']==gates['image_id']==image
assert capture['status']=='FAILED' and capture['commands'][-1]['name']=='quality-check'
assert all(c['status']=='PASS' for c in capture['commands'][:-1])
assert 'Short-path arrays changed; review before release' in (q/'quality-check.log').read_text()
assert replay['status']=='PASS_NEW_IMAGE_MATCHED_STATE_ORIGINAL_TOLERANCES'
assert linear['status']=='PASS_EXPANDED_ROW_PADDING_GRAPH_GATE' and linear['tolerance']==.002
assert len(linear['groups'])==7 and all(len(g['cases'])==18 for g in linear['groups'])
assert gates['status']=='PASS_CHANGED_STACK_LOCAL_GPU_GATES_CAPACITY_SERVING_PENDING'
assert edge['status']=='PASS_FP16_SDPA_EDGE_VALUES_AND_EXISTING_INPUT_GUARDS' and edge['rtol']==.01 and edge['atol']==.003
for pair in d64['comparisons']:
 if pair['other']=='candidate':
  assert all(r['raw']['bitidentical'] and r['processed']['bitidentical'] for r in pair['same_conditioning_rows'])
  assert pair['first_proposal_common_prefix_length']==3
for path in [q/'short/candidate/summary.json',q/'long/candidate/summary.json']:assert json.loads(path.read_text())['api_alignment_verified']
result=dict(status='PASS_REVIEWED_CHANGED_STACK_NUMERICS',image_id=image,
    source_measurement_sha256=sha(q/'changed-stack-quality.json'),failed_exact_campaign_sha256=sha(q/'campaign.json'),
    matched_state_sha256=sha(q/'replay-campaign.json'),linear_gate_sha256=sha(q/'linear/result.json'),
    local_kernel_gate_sha256=sha(root/'E01-gpu-gates/campaign.json'),sdpa_edge_gate_sha256=sha(q/'sdpa-edge-values.json'),
    D64_review_sha256=sha(REPO/'benchmarks/results/b70-stack-20261010/D64-same-prefix-review.json'),
    exact_array_comparison='FAILED_RETAINED_NOT_RELABELLED',numerical_changes=measurement['changes'],
    paired_short_nll_delta_ci95=measurement['paired_short_nll_delta_ci95'],
    rationale='Explicit engineering review of this hashed changed-stack measurement. Short PPL is 3.642167 versus qualified 3.642175 (about -0.0002%); paired mean NLL uncertainty spans zero. BF16-relative mean short KL improves 0.000289 and top1 agreement changes from 490/512 to 492/512, while JS worsens 0.000057, reverse KL and p95 also worsen slightly. These mixed changes are not bit identity or a general quality improvement. The 32K suffix PPL changes from 1.190357 to 1.183419 and all eight BF16-relative top1 probes agree. Independently referenced linear shapes/real weight groups, FP16 SDPA edge cases and all four actual-state attention cases pass their original tolerances. On two actual 64K prefixes, E01 target/sampler distributions and three MTP proposals are bit-identical to qualified v0.31, with independently checked native rejection prefixes. Accept the bounded numerical results as a maintenance candidate; no threshold was widened and the exact check remains failed.',
    limits='Finite panels and four selected actual-state attention cases. Origin of every short-path array difference is not localized; a matching 64K probe does not make all other prefixes identical. Kernel linear references use the existing gated reconstruction plus independent FP32 transforms/matmul. Quality approval alone is not a release: protected serving, capacity/admission, operational and final frozen benchmark gates remain required.')
(q/'numerical-review.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'],flush=True)
