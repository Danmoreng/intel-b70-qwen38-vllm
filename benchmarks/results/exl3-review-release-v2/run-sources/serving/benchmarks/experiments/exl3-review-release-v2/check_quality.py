"""Compact candidate checks using unchanged, already stored reference arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from compare import normalized_logprobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--existing-short',type=Path,required=True)
    p.add_argument('--long-reference',type=Path,required=True);a=p.parse_args()
    comparison=json.loads((a.root/'short/comparison.json').read_text())
    arrays=[]
    for old in sorted(p for p in a.existing_short.glob('window-*.npy') if re.fullmatch(r'window-\d{3}-(nll|logprobs)\.npy',p.name)):
        new=a.root/'short/candidate'/old.name
        # Same checkpoint, profile and short prompt path: report the exact
        # result rather than accepting a degradation using a new PPL cutoff.
        x,y=np.load(old),np.load(new)
        exact=np.array_equal(x,y)
        arrays.append(dict(name=old.name,bitidentical=exact,
            old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(new.read_bytes()).hexdigest()))
    assert len(arrays)==32 and all(x['bitidentical'] for x in arrays), 'Short-path arrays changed; review before release'
    panel=json.loads((a.root/'long/panel.json').read_text());w=panel['windows'][0]
    meta=json.loads((a.root/'long/candidate/summary.json').read_text())
    assert meta['api_alignment_verified'] and len(meta['windows'])==1
    assert meta['panel_sha256']==hashlib.sha256((a.root/'long/panel.json').read_bytes()).hexdigest()
    nll=np.load(a.root/'long/candidate/window-000-nll.npy').astype(np.float64)
    ref=np.load(a.long_reference/'window-000-nll.npy').astype(np.float64)
    nll=nll[w['prefix_tokens']-1:]
    assert nll.shape==ref.shape==(w['suffix_tokens'],) and np.isfinite(nll).all()
    x=normalized_logprobs(np.load(a.long_reference/'window-000-logprobs.npy'))
    y=normalized_logprobs(np.load(a.root/'long/candidate/window-000-logprobs.npy'))
    assert x.shape==y.shape==(len(w['kl_positions']),248320)
    kl=(np.exp(x)*(x-y)).sum(-1);assert np.min(kl)>=-1e-10
    result=dict(status='PASS_COMPACT_REFERENCE_SMOKE',reference_reused=True,
        short=comparison['arms']['candidate']['overall'],short_v1_arrays=arrays,
        long=dict(prefix_tokens=w['prefix_tokens'],suffix_positions=w['suffix_tokens'],
            full_vocabulary_positions=len(kl),original_perplexity=float(np.exp(ref.mean())),
            perplexity=float(np.exp(nll.mean())),kl_mean=float(kl.mean()),
            top1_agreement=float(np.mean(x.argmax(-1)==y.argmax(-1)))),
        scope='Fresh candidate short panel plus first existing 32K-prefix suffix window. Long numerical attention/graph tolerances are separately checked by matched-state replay; not a new broad quality campaign.')
    (a.root/'quality-smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['long']),flush=True)


if __name__=='__main__':main()
