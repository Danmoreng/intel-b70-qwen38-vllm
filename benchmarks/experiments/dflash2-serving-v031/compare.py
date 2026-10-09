#!/usr/bin/env python3
"""Compare only frozen matching C1 requests to the stored qualified v0.31 run."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
REPO=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
receipt=json.loads((REPO/'benchmarks/results/vllm-031-upgrade/serving-summary.json').read_text())
path=REPO/receipt['source_review']['raw_results_path']
assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['source_review']['raw_results_sha256']
reference=json.loads(path.read_text());report=dict(status='EXPERIMENT_ONLY_NOT_PRODUCTION_QUALIFICATION',baseline='stored qualified vLLM 0.31 MTP3, identical frozen first three repeats; no baseline rerun',baseline_sha256=receipt['source_review']['raw_results_sha256'],arms=[])
def summary(cases):
    assert len(cases)==3
    assert all(not c['preemptions'] and c['all_prompt_counts_match'] and c['all_completion_counts_exact'] and c['all_finish_reasons_length'] and c['all_outputs_nonempty'] for c in cases)
    # Native request phase timers do not depend on asynchronously sampled
    # overlapping counter windows. Preserve any window issue; never use its
    # invalid round-equivalent metric in this comparison.
    issues=[dict(repeat=c['repeat'],issues=c['fully_overlapped_counter_window_issues'],residual=c['fully_overlapped_token_accounting_residual']) for c in cases if not c['fully_overlapped_counter_window_valid']]
    return dict(prefill_tps=statistics.median(c['native_prefill_compute_tokens_per_s'] for c in cases),decode_tps=statistics.median(c['native_weighted_decode_tokens_per_s'] for c in cases),ttft_s=statistics.median(q['ttft_s'] for c in cases for q in c['requests']),acceptance=sum(c['speculative_accepted_tokens'] for c in cases)/sum(c['speculative_draft_tokens'] for c in cases),cases=len(cases),sampled_window_issues=issues,round_equivalent_metric_used=False)
for campaign_path in a.campaign:
    campaign=json.loads(campaign_path.read_text());assert campaign['status'] in ['PASS_UNQUANTIZED_DFLASH_STARTUP_ACCEPTANCE_AND_SPEED','PASS_CHECKPOINT_NATIVE_MTP_STARTUP_ACCEPTANCE_AND_SPEED']
    raw_path=REPO/campaign['benchmark_results'];assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==campaign['benchmark_sha256']
    raw=json.loads(raw_path.read_text());arm=dict(method=campaign.get('method','dflash'),name=campaign_path.parent.name,depth=campaign['depth'],context_limit=campaign['active_context_limit'],image=campaign['image'],benchmark_sha256=campaign['benchmark_sha256'],measurements=[],matching_frozen_request_sha256=[])
    for name in sorted(set(c['scenario']['name'] for c in raw['cases'])):
        new=[c for c in raw['cases'] if c['scenario']['name']==name]
        old=[c for c in reference['cases'] if c['scenario']['name']==name and c['repeat'] in [n['repeat'] for n in new]]
        for l,r in zip(old,new):
            assert l['repeat']==r['repeat']
            assert [(q['usage']['prompt_tokens'],q['usage']['completion_tokens']) for q in l['requests']]==[(q['usage']['prompt_tokens'],q['usage']['completion_tokens']) for q in r['requests']]
            assert [q['marker'] for q in l['requests']]==[q['marker'] for q in r['requests']]
            directory=name+'-r'+str(l['repeat'])
            for filename in ['prompt-1.txt','request-1.json']:
                digest=hashlib.sha256((path.parent/directory/filename).read_bytes()).hexdigest()
                assert digest==hashlib.sha256((raw_path.parent/directory/filename).read_bytes()).hexdigest(),filename
                arm['matching_frozen_request_sha256'].append(dict(path=directory+'/'+filename,sha256=digest))
        previous,current=summary(old),summary(new)
        measurement=dict(scenario=name,mtp3=previous,candidate=current,decode_change_pct=(current['decode_tps']/previous['decode_tps']-1)*100)
        if arm['method']=='dflash':measurement['dflash']=current
        arm['measurements'].append(measurement)
    report['arms'].append(arm)
a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
