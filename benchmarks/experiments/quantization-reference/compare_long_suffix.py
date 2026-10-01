"""Compare fixed continuation suffixes with untruncated streamed BF16 contexts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compare import normalized_logprobs, interval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arm',default='target-full-candidate')
    parser.add_argument('--tokenizer',type=Path,required=True)
    args = parser.parse_args()
    root=args.root
    panel=json.loads((root/'panel.json').read_text());digest=sha(root/'panel.json')
    reference=json.loads((root/'bf16/summary.json').read_text())
    candidate=json.loads((root/args.arm/'summary.json').read_text())
    assert reference['panel_sha256']==candidate['panel_sha256']==digest
    assert reference['context_truncated'] is False and candidate['api_alignment_verified'] is True
    n=len(reference['windows'])
    assert len(candidate['windows'])==n==len(panel['windows'])
    tokenizer=json.loads(args.tokenizer.read_text())
    valid=set(tokenizer['model']['vocab'].values())|{r['id'] for r in tokenizer.get('added_tokens',[])}
    assert min(valid)>=0 and max(valid)<248320
    invalid=np.array(sorted(set(range(248320))-valid),dtype=np.int64)
    rows=[];arrays={}
    for index,w in enumerate(panel['windows']):
        first=w['prefix_tokens']-1;count=w['suffix_tokens']
        assert len(w['ids'])-1==first+count
        paths=[root/arm/f'window-{index:03d}-{kind}.npy' for arm in ['bf16',args.arm] for kind in ['nll','logprobs']]
        arrays.update({str(p.relative_to(root)):sha(p) for p in paths})
        ref_nll=np.load(paths[0]).astype(np.float64)
        all_native_nll=np.load(paths[2]).astype(np.float64)
        assert all_native_nll.shape==(len(w['ids'])-1,) and np.isfinite(all_native_nll).all()
        native_nll=all_native_nll[first:first+count]
        assert ref_nll.shape==native_nll.shape==(count,) and np.isfinite(ref_nll).all()
        ref=normalized_logprobs(np.load(paths[1]));other=normalized_logprobs(np.load(paths[3]))
        assert ref.shape==other.shape==(len(w['kl_positions']),248320)
        prob,quant=np.exp(ref),np.exp(other)
        kl=(prob*(ref-other)).sum(-1);reverse=(quant*(other-ref)).sum(-1)
        middle=np.logaddexp(ref,other)-np.log(2)
        js=.5*((prob*(ref-middle)).sum(-1)+(quant*(other-middle)).sum(-1))
        assert min(kl.min(),reverse.min())>=-1e-10
        rows.append({'name':w['name'],'domain':w['domain'],'prefix_tokens':w['prefix_tokens'],
                     'suffix_tokens':count,'original_nll_mean':float(ref_nll.mean()),
                     'original_perplexity':float(np.exp(ref_nll.mean())),
                     'nll_mean':float(native_nll.mean()),'perplexity':float(np.exp(native_nll.mean())),
                     'delta_nll':float((native_nll-ref_nll).mean()),
                     'perplexity_ratio_to_original':float(np.exp((native_nll-ref_nll).mean())),
                     'kl_positions':w['kl_positions'],'kl_original_to_checkpoint':kl.tolist(),
                     'kl_checkpoint_to_original':reverse.tolist(),'js_divergence':js.tolist(),
                     'top1_agreement':float(np.mean(ref.argmax(-1)==other.argmax(-1))),
                     'original_probability_outside_tokenizer_mean':float(prob[:,invalid].sum(-1).mean()),
                     'checkpoint_probability_outside_tokenizer_mean':float(quant[:,invalid].sum(-1).mean())})
    total=sum(r['suffix_tokens'] for r in rows)
    nll=sum(r['nll_mean']*r['suffix_tokens'] for r in rows)/total
    original_nll=sum(r['original_nll_mean']*r['suffix_tokens'] for r in rows)/total
    result={'schema':1,'status':'MEASURED','panel_sha256':digest,'arm':args.arm,
            'reference':reference,'candidate':candidate,'windows':rows,'array_sha256':arrays,
            'overall':{'suffix_positions':total,'kl_positions':sum(len(r['kl_positions']) for r in rows),
                       'original_perplexity':float(np.exp(original_nll)),'perplexity':float(np.exp(nll)),
                       'delta_nll':float(nll-original_nll),
                       'kl_mean':float(np.concatenate([r['kl_original_to_checkpoint'] for r in rows]).mean()),
                       'exploratory_delta_nll_window_ci95':interval([r['delta_nll'] for r in rows])},
            'vocabulary':{'tokenizer_sha256':sha(args.tokenizer),'valid_ids':len(valid),'logit_rows':248320,
                          'masking':'No additional masking in any arm'},
            'scope':panel['scope'],
            'reference_precision_note':'Original BF16 official cached layers, bounded token chunks; chunk-control numerical drift reported separately; not a bit-exact unchunked long reference.',
            'confidence_interval_note':'Exploratory window bootstrap; four engineered windows cannot establish general model quality. No post-hoc pass threshold.'}
    (root/'long-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['overall']),flush=True)


if __name__=='__main__':
    main()
