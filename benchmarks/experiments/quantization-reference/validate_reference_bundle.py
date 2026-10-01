"""Prove local BF16 capture arrays still equal the frozen portable reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    panel_bytes = (root / 'panel.json').read_bytes()
    panel = json.loads(panel_bytes)
    reference = root / 'reference-bf16.npz'
    with reference.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == '41dd8ce312ae11832360649657b936d8a937e8c33bfe6e1382eba083897ef884'
    assert hashlib.sha256(panel_bytes).hexdigest() == 'cbf1a71bbda470859f2c0786cb7134e260111d2072c4753a6732021dadae6171'
    with np.load(reference, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle['metadata_json']))
        assert metadata['panel'] == panel
        assert metadata['reference_summary'] == json.loads((root / 'bf16/summary.json').read_text())
        np.testing.assert_array_equal(bundle['input_ids'], [w['ids'] for w in panel['windows']])
        np.testing.assert_array_equal(bundle['positions'], [w['kl_positions'] for w in panel['windows']])
        distributions = bundle['logprobs']
        nll = bundle['nll']
        assert distributions.shape == (16, 32, 248320)
        assert nll.shape == (16, 1023)
        for i in range(16):
            np.testing.assert_array_equal(distributions[i], np.load(root / 'bf16' / f'window-{i:03d}-logprobs.npy'))
            np.testing.assert_array_equal(nll[i], np.load(root / 'bf16' / f'window-{i:03d}-nll.npy'))
    result = {'schema': 1, 'status': 'PASS', 'bundle_sha256': digest,
              'panel_sha256': hashlib.sha256(panel_bytes).hexdigest(),
              'windows': 16, 'logprobs_shape': [16, 32, 248320],
              'nll_shape': [16, 1023], 'token_ids_and_positions_equal': True,
              'all_distributions_and_nll_bit_exact_to_frozen_bundle': True,
              'summary_equal_to_bundle_metadata': True}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
