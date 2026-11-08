#!/usr/bin/env python3
"""Validate paired frozen payloads and quantify the full profile comparison."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
from statistics import median
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path)
p.add_argument('--out',required=True,type=Path);a=p.parse_args()
def load(root):
 rows={}
 for f in root.glob('*/summary.json'):
  d=json.loads(f.read_text())
  if 'repeat' in d and d['repeat']>=1 and not f.parent.name.startswith('warmup'):
   rows[f.parent.name]=d
 if len(rows)!=70:raise RuntimeError(f'{root}: expected70 waves, got{len(rows)}')
 if sum(len(r['requests']) for r in rows.values())!=124:raise RuntimeError('Expected124 requests')
 for label,r in rows.items():
  for flag in ('all_outputs_nonempty','all_prompt_counts_match','all_completion_counts_exact','all_finish_reasons_length','admission_expectation_met'):
   if not r[flag]:raise RuntimeError(f'{label}: {flag} failed')
  if r['preemptions']!=0:raise RuntimeError(f'{label}: preemption')
  r['request_ttft_median_s']=r['request_ttft_s']['median']
 return rows
old,new=load(a.baseline),load(a.candidate);assert set(old)==set(new)
checks=[];grouped=defaultdict(list)
for label in old:
 x,y=old[label],new[label];assert x['scenario']==y['scenario']
 for i in range(1,len(x['requests'])+1):
  left,right=[p/label/f'prompt-{i}.txt' for p in (a.baseline,a.candidate)]
  assert left.read_bytes()==right.read_bytes(),(label,i,'prompt bytes')
  left,right=[json.loads((p/label/f'request-{i}.json').read_text()) for p in (a.baseline,a.candidate)]
  assert left==right,(label,i,'payload')
  assert x['requests'][i-1]['usage']['prompt_tokens']==y['requests'][i-1]['usage']['prompt_tokens']
 checks.append(label);grouped[x['scenario']['name']].append(label)
keys=('batch_wall_s','native_prefill_compute_tokens_per_s','native_weighted_decode_tokens_per_s',
      'aggregate_decode_wave_tokens_per_s','fully_overlapped_aggregate_decode_tokens_per_s','card_energy_j',
      'request_ttft_median_s','speculative_acceptance','average_card_power_w','prefill_tokens_computed','prompt_tokens_cached',
      'fully_overlapped_round_equivalent_ms','fully_overlapped_speculative_acceptance')
rows=[];triggers=[]
for name,labels in grouped.items():
 r={'scenario':name,'waves':len(labels),'concurrency':old[labels[0]]['scenario']['concurrency'],'metrics':{}}
 for key in keys:
  u=[old[k][key] for k in labels if old[k].get(key) is not None];v=[new[k][key] for k in labels if new[k].get(key) is not None]
  if not u or not v:continue
  x,y=median(u),median(v);change=100*(y/x-1) if x else None
  r['metrics'][key]={'baseline_median':x,'candidate_median':y,'change_pct':change,'baseline_min':min(u),'baseline_max':max(u),'candidate_min':min(v),'candidate_max':max(v)}
 for label,key,sign in [('decode','native_weighted_decode_tokens_per_s',-1),('prefill','native_prefill_compute_tokens_per_s',-1),('wall','batch_wall_s',1)]:
  change=r['metrics'].get(key,{}).get('change_pct')
  if change is not None and sign*change>5:triggers.append({'scenario':name,'metric':label,'change_pct':change})
 rows.append(r)
def overall(data):
 rs=list(data.values());out=sum(x['completion_tokens'] for x in rs);post=sum(x['post_first_completion_tokens'] for x in rs)
 wall=sum(x['batch_wall_s'] for x in rs);energy=sum(x['card_energy_j'] for x in rs)
 return {'waves':len(rs),'requests':sum(len(x['requests']) for x in rs),'output_tokens':out,'batch_wall_s':wall,
  'output_tokens_per_s':out/wall,'card_energy_j':energy,'output_tokens_per_joule':out/energy,
  'weighted_native_decode_tokens_per_s':post/sum(x['native_decode_seconds'] for x in rs),
  'weighted_aggregate_decode_tokens_per_s':post/sum(x['aggregate_decode_window_s'] for x in rs),
  'weighted_native_prefill_tokens_per_s':sum(x['prefill_tokens_computed'] for x in rs)/sum(x['native_prefill_seconds'] for x in rs),
  'preemptions':sum(x['preemptions'] for x in rs),
  'speculative_acceptance':sum(x['speculative_accepted_tokens'] for x in rs)/sum(x['speculative_draft_tokens'] for x in rs),
  'prompt_tokens_cached':sum(x['prompt_tokens_cached'] for x in rs)}
u,v=overall(old),overall(new)
changes={k:100*(v[k]/x-1) for k,x in u.items() if x and k not in ('waves','requests','output_tokens','preemptions')}
for key in ('output_tokens_per_s','weighted_native_decode_tokens_per_s','weighted_native_prefill_tokens_per_s'):
 if changes[key]<-2:triggers.append({'scenario':'complete-matrix','metric':key,'change_pct':changes[key]})
report={'status':'PAIRED_FROZEN_PROFILE_MEASURED_REVIEW_REQUIRED' if triggers else 'PAIRED_FROZEN_PROFILE_NO_THRESHOLD_TRIGGER',
 'baseline':u,'candidate':v,'changes_pct':changes,'scenarios':rows,'investigation_triggers':triggers,
 'matching_sampled_output_sha256':sum(x['output_sha256']==y['output_sha256'] for k in old for x,y in zip(old[k]['requests'],new[k]['requests'])),
 'paired_payloads_and_prompt_bytes_verified':124,'all70_waves_valid_each':True,
 'limits':'Singlehost180W, finite frozen sampled source-review matrix. Investigation thresholds are5%cell or2%weighted; follow-up measurements resolve flagged cases. Generated text and draft acceptance can change across versions.'}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','changes_pct','investigation_triggers')},indent=2))
