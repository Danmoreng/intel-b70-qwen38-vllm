"""Evaluate completed, frozen E01 short-serving evidence without GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
run = args.run.resolve()
gate = json.loads((run / 'short-gate.json').read_text())
manifest = json.loads((run / 'manifest.json').read_text())
assert manifest['blocks'] == 5
rows = [json.loads(line) for line in (run / 'serving-results.jsonl').read_text().splitlines()]
assert len(rows) == 40
for row in rows:
    assert row['completion_tokens'] == 256 and row['preemptions'] == 0
    assert row['counter_delta']['finished_requests'] == 1
    assert row['computed_prompt_tokens'] == row['context'] - row['prefix_cache_hits_delta']
by_case = {}
for row in rows:
    by_case.setdefault((row['block'], row['context'], row['condition']), {})[row['arm']] = row
for case in by_case.values():
    assert set(case) == {'control', 'candidate'}
    assert case['control']['messages_sha256'] == case['candidate']['messages_sha256']
    assert case['control']['computed_prompt_tokens'] == case['candidate']['computed_prompt_tokens']

quality = {arm: json.loads((run / f'block-0-{arm}/quality/results.json').read_text())
           for arm in ('control', 'candidate')}
assert all(q['coding_pass'] for q in quality.values())
for arm in ('control', 'candidate'):
    for check in ('vision-tools', 'prefix-state'):
        assert json.loads((run / f'block-0-{arm}/{check}.json').read_text())['status'] == 'PASS'
nll = []
for control, candidate in zip(quality['control']['nll'], quality['candidate']['nll'], strict=True):
    assert control['status'] == candidate['status'] == 'measured'
    assert control['index'] == candidate['index'] and control['tokens'] == candidate['tokens']
    nll.append({'index': control['index'], 'tokens': control['tokens'],
                'reference': control['nll'], 'candidate': candidate['nll'],
                'increase': candidate['nll'] - control['nll']})
limits = json.loads((run / 'quality-gates.json').read_text())
nll_pass = (statistics.mean(x['increase'] for x in nll) <= limits['teacher_forced_nll_mean_increase_max']
            and all(x['increase'] <= limits['teacher_forced_nll_each_increase_max'] for x in nll))
cells = []
for context in (8192, 16384):
    for cache in ('cold', 'warm'):
        group = [r for r in rows if r['context'] == context and r['condition'] == cache]
        cells.append({'context': context, 'cache': cache,
                      'arm_means': {arm: {metric: statistics.mean(r[metric] for r in group if r['arm'] == arm)
                                         for metric in ('native_prefill_tokens_per_second',
                                                        'native_decode_tokens_per_second',
                                                        'mtp_acceptance_rate')}
                                    for arm in ('control', 'candidate')}})
output = {'short_serving_pass': gate['pass'], 'nll_pass': nll_pass, 'nll': nll,
          'nll_mean_increase': statistics.mean(x['increase'] for x in nll),
          'coding': '3/3 each arm', 'vision_tools_prefix': 'PASS each arm', 'cells': cells,
          'paired_latency_results': json.loads((run / 'paired-summary.json').read_text()),
          'scope': '8K/16K short serving only. Operator-region gains are not serving gains.',
          'long_context_qualified': False, 'production_promoted': False,
          'evidence_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (run/'serving-results.jsonl', run/'paired-summary.json',
                                        run/'fixtures.json', run/'context-benchmark.py')}}
(run / 'evaluation.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({k: v for k, v in output.items() if k not in ('paired_latency_results', 'evidence_sha256')}, indent=2))
