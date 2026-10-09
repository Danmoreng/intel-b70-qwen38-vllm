#!/usr/bin/env python3
"""Publish only complete new-image gates and compare stored v2 measurements."""
import argparse,hashlib,importlib.util,json,statistics
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
root=REPO/'benchmark-results/vllm-031-upgrade-20261009';cfg=REPO/'config/experiments/vllm-031-upgrade';sources={}
def read(p):
 p=Path(p);sources[str(p.relative_to(REPO))]=sha(p);return json.loads(p.read_text())
r=read(cfg/'production_image.json');image=r['image_id'];policy=r['policy_sha256'];assert sha(cfg/'production_policy.json')==policy
kernels=read(root/'kernels-final/campaign.json');quality=read(root/'quality-final/numerical-review.json');replay=read(root/'quality-final/replay-campaign.json');operations=read(root/'operations-final/operations.json');serving=read(root/'serving-final/campaign.json')
for d in [kernels,quality,replay,operations]:assert d['image_id']==image
assert kernels['status']=='PASS_NEW_ABI_KERNEL_GATES' and all(d['status']=='PASS' for d in kernels['checks'].values())
assert quality['status']=='PASS_REVIEWED_CROSS_VERSION_NUMERICS' and quality['policy_sha256']==policy
assert replay['status']=='PASS_NEW_IMAGE_MATCHED_STATE_ORIGINAL_TOLERANCES'
assert operations['status']=='PASS_NEW_IMAGE_OPERATIONAL_GATES' and operations['policy_sha256']==policy
assert serving['status']=='COMPLETE_REVIEW_SERVING_MATRIX' and serving['image_receipt']['image_id']==image
coverage=read(root/'operations-final/runtime-sources/SOURCE_GUARD_COVERAGE.json');assert coverage['required_count']==coverage['covered_count']==16
for name,h in coverage['files_sha256'].items():assert sha(root/'operations-final/runtime-sources'/name)==h
assert read(root/'operations-final/runtime-sources/CAPTURE_IDENTITY.json')['image_id']==image
boundary=operations['cases']['near-limit'];assert boundary['usage']['prompt_tokens']==261120 and boundary['usage']['completion_tokens']==1024 and boundary['native']['preemptions']==0
assert operations['cases']['c4-long']['native']['preemptions']==0 and operations['confirmed_abort_trace_occurrences']>0 and operations['cases']['worker-restart']['status']=='PASS'
steps=[]
for path in sorted((root/'quality-final/replay/observations/steps').glob('*-result.json')):
 d=read(path);assert d['status']=='PASS_LOCAL_ATTENTION_ORIGINAL_TOLERANCE'
 assert all(x['local_original_tolerance_pass'] and x['shape'][1]==248320 and x['initial_state_verified_sha256']==d['restored_state_sha256'] for x in d['routes'].values())
 steps.append(dict(label=d['label'],graph_mode=d['graph_mode'],rows_by_request=d['rows_by_request'],local_attention_comparisons=sum(len(x['local_attention_checks']) for x in d['routes'].values())))
assert len(steps)==4 and any(d['graph_mode']=='FULL' for d in steps) and any(d['label']=='long-prefill' for d in steps)
spec=importlib.util.spec_from_file_location('upgrade_serving_export',REPO/'scripts/summarize-readme-benchmarks.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
summary=m.summarize(root/'serving-final',serving_only=True);assert summary['image_id']==image and summary['source_review']['waves']==70 and summary['source_review']['requests']==124
(out/'serving-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
old=read(REPO/'benchmarks/results/exl3-review-release-v2/serving-summary.json');old_release=read(REPO/'config/releases/exl3-v2/production_image.json');assert old['image_id']==old_release['image_id'] and old['power_cap_w']==summary['power_cap_w']==180
previous={d['name']:d for d in old['source_review']['scenario_results']};rows=[]
raw_old=read(REPO/old['source_review']['raw_results_path']);raw_new=read(REPO/summary['source_review']['raw_results_path'])
for new in summary['source_review']['scenario_results']:
 before=previous[new['name']];row=dict(name=new['name'],concurrency=new['concurrency'])
 for key in ['prefill_tps_median','decode_tps_median','ttft_s_median','batch_wall_s_median','fully_overlapped_decode_tps','mtp_acceptance']:
  x,y=before.get(key),new.get(key);row[key]=dict(stored_v2=x,v031=y,change_pct=100*(y/x-1) if x is not None and y is not None and x!=0 else None)
 for label,raw in [('stored_v2',raw_old),('v031',raw_new)]:
  waves=[x for x in raw['cases'] if x['scenario']['name']==new['name']]
  valid=[x['fully_overlapped_round_equivalent_ms'] for x in waves if x['fully_overlapped_counter_window_valid']]
  row.setdefault('round_equivalent_ms',{})[label]=statistics.median(valid) if valid else None
  row.setdefault('paired_waves',{})[label]=[dict(repeat=x['repeat'],decode_tps=x['native_weighted_decode_tokens_per_s'],mtp_acceptance=x['speculative_acceptance'],round_equivalent_ms=x['fully_overlapped_round_equivalent_ms'],counter_window_valid=x['fully_overlapped_counter_window_valid']) for x in waves]
 x,y=row['round_equivalent_ms']['stored_v2'],row['round_equivalent_ms']['v031'];row['round_equivalent_ms']['change_pct']=100*(y/x-1) if x and y else None
 row['review_trigger']=bool((row['decode_tps_median']['change_pct'] or 0)<-5 or (row['ttft_s_median']['change_pct'] or 0)>5)
 rows.append(row)
comparison=dict(status='COMPLETE_STORED_BASELINE_COMPARISON',baseline_regenerated=False,baseline_image_id=old['image_id'],candidate_image_id=image,baseline_summary_sha256=sha(REPO/'benchmarks/results/exl3-review-release-v2/serving-summary.json'),candidate_summary_sha256=sha(out/'serving-summary.json'),scenarios=rows,scope='Same frozen prompt bytes, sampling values, power cap and serving contract; a stored historical comparison is not a contemporaneous paired A/B or a statistical guarantee. Five-percent changes are review triggers, not correctness thresholds.')
(out/'benchmark-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
assessment=dict(status='PASS_EXACT_IMAGE_RELEASE_GATES',image_id=image,policy_sha256=policy,source_commit=r['source_commit'],runtime_artifacts_sha256=r['runtime_artifacts_sha256'],kernel_gates=kernels,numerical_review=quality,quality_measurements=read(root/'quality-final/cross-version-quality.json'),state_replays=steps,operations=operations,source_guard_coverage=dict(required=16,covered=16),serving=dict(scenarios=20,waves=70,requests=124,preemptions=0),receipts_sha256=sources,benchmark_comparison_sha256=sha(out/'benchmark-comparison.json'),limitations=['Full-reference output distributions change across Torch/vLLM versions; old bitidentity gate failed and is retained, not relabeled. New-image numerical gates retain original tolerances.','Frozen finite quality panels and state probes cannot guarantee identical generated text or all-task accuracy.','Official upstream dependency metadata conflicts are inherited unchanged; actual GPU ABI and serving tests qualify this exact stack.','Clean C4 and maximum single context are separate from C16 pressure, which permits recorded preemptions.','DFlash, old-runtime inference and old-baseline benchmark reruns were excluded.'])
(out/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
triggers=[row['name'] for row in rows if row['review_trigger']]
decision=dict(status='REQUIRES_PERFORMANCE_REVIEW' if triggers else 'QUALIFIED_FOR_PRODUCTION_PROMOTION',image_id=image,policy_sha256=policy,source_commit=r['source_commit'],gate_assessment=dict(path=str((out/'assessment.json').relative_to(REPO)),sha256=sha(out/'assessment.json')),benchmark_summary=dict(path=str((out/'serving-summary.json').relative_to(REPO)),sha256=sha(out/'serving-summary.json')),immediate_rollback_image_id=old_release['image_id'],performance_review_triggers=triggers,rationale='vLLM 0.31/Torch 2.14 XPU upgrade: rebuilt C++20 EXL3/M04, guarded upstream integration, actual-state original-tolerance correctness, reviewed cross-version quality changes, full context/API/media/abort/restart contract and complete frozen serving benchmark qualified. Original references and historical v2 benchmark reused; no old baseline rerun.')
(out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(dict(status=decision['status'],image_id=image,performance_review_triggers=triggers),indent=2))
