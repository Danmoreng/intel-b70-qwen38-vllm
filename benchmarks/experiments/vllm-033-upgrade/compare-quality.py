#!/usr/bin/env python3
"""Paired finite teacher-forced measurements; no automatic quality approval."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path)
p.add_argument('--out',required=True,type=Path);a=p.parse_args()
result={'status':'MEASURED_PAIRED_FINITE_QUALITY','baseline':str(a.baseline),'candidate':str(a.candidate),
        'limits':'Finite teacher-forced panels, not coding-task success or general text equivalence. Confidence intervals resample paired windows, not independent tokens.'}
rng=np.random.default_rng(20261108)
for kind in ('short','long'):
 panel=json.loads((a.baseline/kind/'panel.json').read_text());other=json.loads((a.candidate/kind/'panel.json').read_text())
 assert panel==other
 old_meta=json.loads((a.baseline/kind/'capture/summary.json').read_text())
 new_meta=json.loads((a.candidate/kind/'capture/summary.json').read_text())
 assert old_meta['api_alignment_verified'] and new_meta['api_alignment_verified']
 assert old_meta['panel_sha256']==new_meta['panel_sha256']==hashlib.sha256((a.baseline/kind/'panel.json').read_bytes()).hexdigest()
 n=old_meta['windows'];assert len(n)==len(new_meta['windows'])
 rows=[];old_losses=[];new_losses=[];full_losses=[];identity=[]
 for i,_ in enumerate(n):
  path=f'window-{i:03d}'
  x0=np.load(a.baseline/kind/'capture'/(path+'-nll.npy'));y0=np.load(a.candidate/kind/'capture'/(path+'-nll.npy'))
  assert x0.shape==y0.shape and np.isfinite(x0).all() and np.isfinite(y0).all()
  full_losses.append((x0.astype(np.float64),y0.astype(np.float64)))
  if kind=='long':
   start=panel['windows'][i]['prefix_tokens']-1
   x0=x0[start:];y0=y0[start:]
  x,y=x0.astype(np.float64),y0.astype(np.float64)
  old_losses.append(x);new_losses.append(y)
  old_lp=np.load(a.baseline/kind/'capture'/(path+'-logprobs.npy'))
  new_lp=np.load(a.candidate/kind/'capture'/(path+'-logprobs.npy'))
  assert old_lp.shape==new_lp.shape and old_lp.shape[1]==248320
  assert np.isfinite(old_lp).all() and np.isfinite(new_lp).all()
  exact=np.array_equal(x0,y0) and np.array_equal(old_lp,new_lp);identity.append(exact)
  u,v=old_lp.astype(np.float64),new_lp.astype(np.float64)
  u-=np.logaddexp.reduce(u,axis=-1,keepdims=True);v-=np.logaddexp.reduce(v,axis=-1,keepdims=True)
  kl=(np.exp(u)*(u-v)).sum(-1);assert kl.min()>=-1e-10
  rows.append({'window':i,'domain':panel['windows'][i]['domain'],'positions':len(x),
   'full_vocabulary_positions':len(kl),'baseline_nll':float(x.mean()),'candidate_nll':float(y.mean()),
   'nll_delta':float((y-x).mean()),'full_prompt_positions':len(full_losses[-1][0]),
   'full_prompt_nll_delta':float((full_losses[-1][1]-full_losses[-1][0]).mean()),'old_to_new_kl_mean':float(kl.mean()),
   'top1_matches':int((u.argmax(-1)==v.argmax(-1)).sum()),'bitidentical':exact})
 old=np.concatenate(old_losses);new=np.concatenate(new_losses)
 full_old=np.concatenate([x for x,y in full_losses]);full_new=np.concatenate([y for x,y in full_losses])
 deltas=np.array([r['nll_delta'] for r in rows]);ci=None
 if len(rows)>1:ci=np.quantile(deltas[rng.integers(0,len(rows),(10000,len(rows)))].mean(-1),[.025,.975]).tolist()
 result[kind]={'windows':rows,'positions':len(old),
  'loss_scope':'last128suffixpositions per long window' if kind=='long' else 'all prompt positions',
  'full_prompt_positions':len(full_old),'full_prompt_nll_delta':float((full_new-full_old).mean()),
  'full_prompt_perplexity_change_pct':float(100*np.expm1(full_new.mean()-full_old.mean())),
  'baseline_perplexity':float(np.exp(old.mean())),
  'candidate_perplexity':float(np.exp(new.mean())),'perplexity_change_pct':float(100*np.expm1(new.mean()-old.mean())),
  'paired_nll_delta':float((new-old).mean()),'paired_window_nll_delta_ci95':ci,
  'top1_matches':sum(r['top1_matches'] for r in rows),
  'full_vocabulary_positions':sum(r['full_vocabulary_positions'] for r in rows),'all_arrays_bitidentical':all(identity)}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({kind:{k:v for k,v in result[kind].items() if k!='windows'} for kind in ('short','long')},indent=2))
