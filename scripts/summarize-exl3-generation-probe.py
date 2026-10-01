#!/usr/bin/env python3
"""Audit generated argmax choices and retain shared-prefix divergence evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def divergence(a, b):
    position = next((i for i, (x, y) in enumerate(zip(a['token_ids'], b['token_ids'])) if x != y), None)
    if position is None:
        return {'first_difference': None, 'compared_tokens': len(a['token_ids'])}
    sa, sb = a['logprobs'][position], b['logprobs'][position]
    common = set(item['id'] for item in sa['top']) & set(item['id'] for item in sb['top'])
    scores_a = {item['id']: item['logprob'] for item in sa['top']}
    scores_b = {item['id']: item['logprob'] for item in sb['top']}
    return {'first_difference': position, 'token_a': a['token_ids'][position], 'token_b': b['token_ids'][position],
            'scores_a': sa, 'scores_b': sb, 'common_top_logprob_ids': sorted(common),
            'max_common_top_logprob_difference': max((abs(scores_a[t] - scores_b[t]) for t in common), default=None),
            'scope': 'Predictions share exactly the generated history before this position. Top5 intersections are not full-vocabulary KL. No accepted error threshold is inferred from an observed margin.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); root = args.campaign.resolve()
    state = json.loads((root / 'campaign.json').read_text()); assert state['status'] == 'COMPLETE'
    assert [a['variant'] for a in state['arms']] == ['graph-mtp3', 'graph-mtp4', 'eager-mtp3', 'graph-nospec']
    values, audits = {}, []
    for arm in state['arms']:
        path = root / arm['variant'] / 'summary.json'; data = json.loads(path.read_text())
        assert arm['status'] == data['status'] == 'COMPLETE' and sha(path) == arm['summary_sha256']
        assert data['panel_sha256'] == state['panel_sha256'] and data['engine_config'] == arm['engine_config']
        assert len(data['waves']) == 8 and len(data['forced_next_token']) == 4
        tokens = non_argmax = 0
        for wave in data['waves']:
            assert len(wave['responses']) == wave['concurrency']
            budget = 192 if wave['window'] == 'code-4096' else 64
            for response in wave['responses']:
                assert len(response['token_ids']) == len(response['logprobs']) == budget
                assert hashlib.sha256(json.dumps(response['token_ids'], separators=(',', ':')).encode()).hexdigest() == response['output_ids_sha256']
                tokens += budget
                non_argmax += sum(score['chosen_gap_from_top'] != 0 for score in response['logprobs'])
        assert tokens == 2560
        graph_modes = Counter(batch['target_graph_mode'] for wave in data['waves'] for frames in wave['batches']
                              for batch in frames if not batch['has_prefill'])
        audits.append({'variant': arm['variant'], 'summary_sha256': sha(path), 'generated_tokens': tokens,
                       'choices_below_reported_top_score': non_argmax, 'pure_decode_graph_modes': dict(graph_modes)})
        values[arm['variant']] = data
    base = values['graph-mtp3']; comparisons = []
    for variant, data in values.items():
        if variant == 'graph-mtp3': continue
        pairs = []
        for a, b in zip(base['waves'], data['waves']):
            assert (a['window'], a['concurrency'], a['cache_mode']) == (b['window'], b['concurrency'], b['cache_mode'])
            for index, (ra, rb) in enumerate(zip(a['responses'], b['responses'])):
                pairs.append({'window': a['window'], 'concurrency': a['concurrency'], 'cache_mode': a['cache_mode'],
                              'request': index, **divergence(ra, rb)})
        forced = []
        for a, b in zip(base['forced_next_token'], data['forced_next_token']):
            assert a['prompt_ids_sha256'] == b['prompt_ids_sha256']
            forced.append({'window': a['window'], 'position': a['position'], 'prompt_ids_sha256': a['prompt_ids_sha256'],
                           'expected_reference_token': a['reference_expected_token'],
                           'only_prefill_target_calls_a': all(batch['has_prefill'] for frames in a['batches'] for batch in frames),
                           'only_prefill_target_calls_b': all(batch['has_prefill'] for frames in b['batches'] for batch in frames),
                           **divergence(a['response'], b['response'])})
        comparisons.append({'variant': variant, 'paired_requests': len(pairs),
                            'exact_matches': sum(p['first_difference'] is None for p in pairs),
                            'pairs': pairs, 'identical_prefix_one_token_controls': forced})
    result = {'status': 'COMPLETE_BOUNDED_DIAGNOSTIC_QUALITY_GATE_REMAINS_PARTIAL',
              'image_id': state['image_id'], 'campaign_sha256': sha(root / 'campaign.json'),
              'source_sha256': state['source_sha256'], 'audits': audits, 'comparisons': comparisons,
              'interpretation': 'All10240 main generated tokens select a reported maximum-score token. Different routes/batching/cache reuse change scores and can flip rankings. Some flips persist in one-token forced-prefix prefill controls, so they are not specific proof of speculative verification failure. No full-vocabulary KL or BF16 generated-path equivalence is established. Do not label all changes harmless rounding or a sampler bug.'}
    assert not args.out.exists(); args.out.mkdir(parents=True)
    (args.out / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'non_argmax': sum(a['choices_below_reported_top_score'] for a in audits),
                      'matches': {r['variant']: r['exact_matches'] for r in comparisons}}))


if __name__ == '__main__':
    main()
