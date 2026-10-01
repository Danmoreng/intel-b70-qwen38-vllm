"""Quantify chunked BF16 reference drift on two unchanged original short windows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compare import normalized_logprobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    panel = json.loads((args.root/'panel.json').read_text())
    original = json.loads((args.original/'panel.json').read_text())
    rows=[]
    for index,window in enumerate(panel['windows']):
        source = window['source_window']
        old = original['windows'][source]
        assert window['ids']==old['ids']
        first = window['prefix_tokens']-1
        reference_nll = np.load(args.original/'bf16'/f'window-{source:03d}-nll.npy')[first:]
        current_nll = np.load(args.root/'bf16'/f'window-{index:03d}-nll.npy')
        assert reference_nll.shape == current_nll.shape == (window['suffix_tokens'],)
        reference_lp = np.load(args.original/'bf16'/f'window-{source:03d}-logprobs.npy')
        selected = [old['kl_positions'].index(p) for p in window['kl_positions']]
        ref = normalized_logprobs(reference_lp[selected])
        new = normalized_logprobs(np.load(args.root/'bf16'/f'window-{index:03d}-logprobs.npy'))
        assert ref.shape == new.shape
        kl = (np.exp(ref)*(ref-new)).sum(-1)
        assert kl.min() >= -1e-10
        rows.append({'name':window['name'],'suffix_tokens':window['suffix_tokens'],
                     'kl_positions':window['kl_positions'],'kl_mean':float(kl.mean()),'kl_max':float(kl.max()),
                     'delta_nll':float(np.mean(current_nll.astype(np.float64)-reference_nll)),
                     'maximum_target_nll_difference':float(np.max(np.abs(current_nll-reference_nll))),
                     'top1_agreement':float(np.mean(ref.argmax(-1)==new.argmax(-1)))})
    result={'schema':1,'status':'MEASURED','panel_sha256':hashlib.sha256((args.root/'panel.json').read_bytes()).hexdigest(),
            'windows':rows,'note':'Numerical reference drift, not a new quantization quality score or a post-hoc release threshold'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
