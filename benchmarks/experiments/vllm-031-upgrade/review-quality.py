"""Measure cross-version changes without relabeling old bit-exact checks."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from compare import normalized_logprobs
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
old=Path('/old-short');new=root/'short/candidate'
reference=json.loads(Path('/old-assessment.json').read_text())['quality_smoke']
short=json.loads((root/'short/comparison.json').read_text())['arms']['candidate']['overall']
arrays=[];windows=[]
for f in sorted(old.glob('window-*.npy')):
 if not re.fullmatch(r'window-\d{3}-(nll|logprobs)\.npy',f.name):continue
 x,y=np.load(f),np.load(new/f.name);assert x.shape==y.shape and np.isfinite(y).all()
 arrays.append(dict(name=f.name,bitidentical=np.array_equal(x,y),old_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),new_sha256=hashlib.sha256((new/f.name).read_bytes()).hexdigest(),max_absolute=float(np.max(np.abs(x.astype('float64')-y)))))
for i in range(16):
 x=normalized_logprobs(np.load(old/f'window-{i:03d}-logprobs.npy'));y=normalized_logprobs(np.load(new/f'window-{i:03d}-logprobs.npy'))
 kl=(np.exp(x)*(x-y)).sum(-1);js=.5*((np.exp(x)*(x-np.logaddexp(x,y)+np.log(2))).sum(-1)+(np.exp(y)*(y-np.logaddexp(x,y)+np.log(2))).sum(-1))
 windows.append(dict(window=i,old_to_new_kl_mean=float(kl.mean()),js_mean=float(js.mean()),top1_agreement=float(np.mean(x.argmax(-1)==y.argmax(-1))),nll_delta=float(np.load(new/f'window-{i:03d}-nll.npy').astype('float64').mean()-np.load(old/f'window-{i:03d}-nll.npy').astype('float64').mean())))
w=json.loads((root/'long/panel.json').read_text())['windows'][0]
x=normalized_logprobs(np.load('/long-reference/window-000-logprobs.npy'));y=normalized_logprobs(np.load(root/'long/candidate/window-000-logprobs.npy'))
nll=np.load(root/'long/candidate/window-000-nll.npy').astype('float64')[w['prefix_tokens']-1:];ref=np.load('/long-reference/window-000-nll.npy').astype('float64');assert nll.shape==ref.shape==(w['suffix_tokens'],)
long=dict(prefix_tokens=w['prefix_tokens'],suffix_positions=len(nll),full_vocabulary_positions=len(x),original_perplexity=float(np.exp(ref.mean())),perplexity=float(np.exp(nll.mean())),kl_mean=float((np.exp(x)*(x-y)).sum(-1).mean()),top1_agreement=float(np.mean(x.argmax(-1)==y.argmax(-1))))
rng=np.random.default_rng(20261009);d=np.array([x['nll_delta'] for x in windows]);ci=np.quantile(d[rng.integers(0,16,(10000,16))].mean(-1),[.025,.975]).tolist()
d=dict(status='MEASURED_CROSS_VERSION_CHANGE_REQUIRES_NUMERICAL_REVIEW',reference_reused=True,short=short,long=long,stored_v2=reference,short_arrays=arrays,cross_version_windows=windows,paired_short_nll_delta_ci95=ci,changes=dict(short_ppl_pct=100*(short['perplexity']/reference['short']['perplexity']-1),short_kl_delta=short['kl_mean']-reference['short']['kl_mean'],short_js_delta=short['js_mean']-reference['short']['js_mean'],long_ppl_pct=100*(long['perplexity']/reference['long']['perplexity']-1),long_kl_delta=long['kl_mean']-reference['long']['kl_mean']),scope='New Torch/vLLM changes cannot be approved by the old same-runtime bitidentity check. The failed old check remains evidence; review these differences together with unchanged FP32/native kernel and actual-state numerical gates. No old runtime was executed.')
(root/'cross-version-quality.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:d[k] for k in ['status','short','long','changes','paired_short_nll_delta_ci95']},indent=2))
