"""Freeze four long prefix/suffix probes from already frozen source token IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-context', type=int, default=262144)
    parser.add_argument('--control', action='store_true', help='Two original 1024-token windows to compare chunked BF16 with existing BF16')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to replace a frozen panel')
    data = args.input.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == 'cbf1a71bbda470859f2c0786cb7134e260111d2072c4753a6732021dadae6171'
    old = json.loads(data)
    windows = []
    if args.control:
        for source_index in (0, 8):
            source = old['windows'][source_index]
            windows.append({**source, 'source_window': source_index,
                            'prefix_tokens': 768, 'suffix_tokens': 256,
                            'kl_positions': [p for p in source['kl_positions'] if p >= 767]})
    else:
        for index, (prefix, domain) in enumerate(((32768, 'code'), (102400, 'prose'),
                                                (184320, 'code'), (args.maximum_context - 129, 'prose'))):
            source_indices = [i for i, w in enumerate(old['windows']) if w['domain'] == domain]
            material = [token for i in source_indices for token in old['windows'][i]['ids']]
            ids = (material * ((prefix + len(material) - 1) // len(material)))[:prefix]
            suffix_source = old['windows'][source_indices[(index + 1) % len(source_indices)]]['ids']
            ids += suffix_source[257:385]
            window = {'name': f'{domain}-prefix{prefix}-suffix128', 'domain': domain,
                      'ids': ids, 'prefix_tokens': prefix, 'suffix_tokens': 128,
                      'kl_positions': sorted(random.Random(20261001 + index).sample(range(prefix - 1, prefix + 127), 8)),
                      'prefix_source_windows': source_indices,
                      'suffix_source_window': source_indices[(index + 1) % len(source_indices)],
                      'suffix_source_token_range': [257, 385]}
            window['token_ids_sha256'] = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            windows.append(window)
    result = {'schema': 1, 'seed': 20261001, 'source_panel_sha256': digest,
              'original_revision': old['original_revision'], 'maximum_context': args.maximum_context,
              'control': args.control, 'prediction_alignment': 'logit at prefix-1 predicts first fixed suffix token',
              'scope': 'Precision drift probes with repeated frozen code/prose prefixes; not a natural long-document or agent-quality benchmark',
              'windows': windows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(json.dumps({'panel_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'windows': [{k:w[k] for k in ('name', 'prefix_tokens', 'suffix_tokens')} for w in windows]}), flush=True)


if __name__ == '__main__':
    main()
