#!/usr/bin/env python3
"""Validate complete public runs, publish compact counters, and refresh README."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def summarize(root):
    state = json.loads((root / 'state.json').read_text())
    if state['status'] != 'complete':
        raise RuntimeError('benchmark has not completed')
    release = json.loads((REPO / 'config/production_image.json').read_text())
    live = json.loads((root / 'production-identity.json').read_text())
    restored = json.loads((root / 'production-serving.json').read_text())
    if live != restored or live['image_id'] != release['image_id'] or live['policy_sha256'] != release['policy_sha256']:
        raise RuntimeError('benchmark image/policy identity changed')
    paths = list((root / 'source-review').glob('*/results.json'))
    if len(paths) != 1:
        raise RuntimeError('expected one complete source-review run')
    path = paths[0]
    data = json.loads(path.read_text())
    cases = data['cases']
    plan = json.loads((REPO / 'benchmarks/current-profile-scenarios.json').read_text())
    expected = {(row['name'], repeat): row for row in plan for repeat in range(1, row['repeats'] + 1)}
    if len(cases) != len(expected) or {(row['scenario']['name'], row['repeat']) for row in cases} != set(expected):
        raise RuntimeError('scenario/wave coverage differs from the frozen full plan')
    fixture_hashes = json.loads((root / 'fixture-sha256.json').read_text())
    manifest = data['manifest']
    groups = {}
    for case in cases:
        scenario = case['scenario']
        if any(scenario[key] != expected[scenario['name'], case['repeat']][key]
               for key in ('prompt_tokens', 'output_tokens', 'concurrency', 'repeats')):
            raise RuntimeError('scenario budgets changed')
        if len(case['requests']) != scenario['concurrency']:
            raise RuntimeError('request coverage differs')
        if not all(case[key] for key in ('all_prompt_counts_match', 'all_completion_counts_exact',
                                         'all_finish_reasons_length', 'all_outputs_nonempty')):
            raise RuntimeError('request/token/finish validation failed')
        if case['preemptions'] or case['prefill_recompute_excess']:
            raise RuntimeError('preempted or recomputed work requires review')
        folder = path.parent / f"{scenario['name']}-r{case['repeat']}"
        for index, request in enumerate(case['requests'], 1):
            if request['usage']['completion_tokens'] != scenario['output_tokens'] or request['finish_reason'] != 'length':
                raise RuntimeError('output count differs')
            for filename in (f'prompt-{index}.txt', f'request-{index}.json'):
                file = folder / filename
                relative = str(file.relative_to(path.parent))
                if sha(file) != fixture_hashes[relative]:
                    # The replay reconstructs messages after loading metadata,
                    # so JSON key order can differ without changing the payload.
                    reference = Path(manifest['fixture_root']) / relative
                    if (not filename.endswith('.json') or sha(reference) != fixture_hashes[relative]
                            or json.loads(file.read_text()) != json.loads(reference.read_text())):
                        raise RuntimeError('frozen prompt/request differs: ' + relative)
        groups.setdefault(scenario['name'], []).append(case)
    scenario_results = []
    for scenario in plan:
        rows = groups[scenario['name']]
        drafted = sum(row['speculative_draft_tokens'] for row in rows)
        elapsed = sum(row['fully_overlapped_decode_sampled_s'] or 0 for row in rows)
        generated = sum(row['fully_overlapped_generation_tokens'] or 0 for row in rows)
        counts = [request['usage']['prompt_tokens'] for row in rows for request in row['requests']]
        scenario_results.append({
            'name': scenario['name'], 'group': scenario['group'], 'concurrency': scenario['concurrency'],
            'waves': len(rows), 'actual_prompt_tokens_min': min(counts), 'actual_prompt_tokens_max': max(counts),
            'prefill_tps_median': median(rows, 'native_prefill_compute_tokens_per_s'),
            'decode_tps_median': median(rows, 'native_weighted_decode_tokens_per_s'),
            'mtp_acceptance': sum(row['speculative_accepted_tokens'] for row in rows) / drafted,
            'ttft_s_median': statistics.median(request['ttft_s'] for row in rows for request in row['requests']),
            'batch_wall_s_median': median(rows, 'batch_wall_s'),
            'fully_overlapped_decode_tps': generated / elapsed if elapsed else None,
            'max_waiting': max(row['peak_waiting'] for row in rows),
            'preemptions': sum(row['preemptions'] for row in rows),
            'round_equivalent_valid_waves': sum(row['fully_overlapped_counter_window_valid'] for row in rows),
            'round_equivalent_invalid_waves': sum(not row['fully_overlapped_counter_window_valid'] for row in rows),
        })
    prefix = []
    for row in cases:
        if row['scenario']['group'] == 'prefix-cache':
            prefix.append({'name': row['scenario']['name'], 'repeat': row['repeat'],
                           'prompt_tokens': row['requests'][0]['usage']['prompt_tokens'],
                           'cached_tokens': row['prompt_tokens_cached'], 'computed_tokens': row['prefill_tokens_computed'],
                           'ttft_s': row['requests'][0]['ttft_s'], 'wall_s': row['requests'][0]['wall_s']})
    full_profile_prefix = list(prefix)
    supplementary = []
    if next(row for row in prefix if row['name'] == 'prefix-64k-cold-warm')['cached_tokens']:
        isolated = root / 'prefix-64k-isolated'
        if json.loads((isolated / 'state.json').read_text())['status'] != 'complete':
            raise RuntimeError('isolated cold 64K prefix run has not completed')
        separate_live = json.loads((isolated / 'production-identity.json').read_text())
        if any(separate_live[key] != live[key] for key in ('image_id', 'image_tag', 'policy_sha256', 'command')):
            raise RuntimeError('isolated prefix used a different image or profile')
        separate_paths = list((isolated / 'source-review').glob('*/results.json'))
        if len(separate_paths) != 1:
            raise RuntimeError('expected one isolated prefix run')
        separate_path = separate_paths[0]
        separate = json.loads(separate_path.read_text())
        if [row['repeat'] for row in separate['cases']] != [1, 2, 3]:
            raise RuntimeError('isolated prefix wave coverage differs')
        replacement = []
        for row in separate['cases']:
            name = row['scenario']['name']
            if name != 'prefix-64k-cold-warm' or len(row['requests']) != 1:
                raise RuntimeError('unexpected isolated prefix scenario')
            if row['preemptions'] or row['prefill_recompute_excess'] or not all(row[key] for key in ('all_prompt_counts_match', 'all_completion_counts_exact', 'all_finish_reasons_length', 'all_outputs_nonempty')):
                raise RuntimeError('isolated prefix request validation failed')
            folder = separate_path.parent / f"{name}-r{row['repeat']}"
            original_folder = Path(manifest['fixture_root']) / folder.name
            if ((folder / 'prompt-1.txt').read_bytes() != (original_folder / 'prompt-1.txt').read_bytes()
                    or json.loads((folder / 'request-1.json').read_text()) != json.loads((original_folder / 'request-1.json').read_text())):
                raise RuntimeError('isolated prefix fixtures differ')
            replacement.append({'name': name, 'repeat': row['repeat'],
                'prompt_tokens': row['requests'][0]['usage']['prompt_tokens'],
                'cached_tokens': row['prompt_tokens_cached'], 'computed_tokens': row['prefill_tokens_computed'],
                'ttft_s': row['requests'][0]['ttft_s'], 'wall_s': row['requests'][0]['wall_s']})
        prefix = [row for row in prefix if row['name'] != 'prefix-64k-cold-warm'] + replacement
        supplementary.append({'name': 'prefix-64k-cold-warm', 'run_id': separate_path.parent.name,
            'raw_results_path': str(separate_path.relative_to(REPO)), 'raw_results_sha256': sha(separate_path),
            'image_id': separate_live['image_id'], 'requests': 3,
            'started_at': separate['manifest']['started_at'],
            'finished_at': json.loads((isolated / 'state.json').read_text())['finished_at'],
            'reason': 'The complete run reused 13312 prefix tokens from its preceding 16K requests. A fresh worker provides the cold 64K observation.'})
    for name in ('prefix-16k-cold-warm', 'prefix-64k-cold-warm'):
        rows = [row for row in prefix if row['name'] == name]
        if rows[0]['cached_tokens'] != 0 or any(row['cached_tokens'] <= 0 for row in rows[1:]):
            raise RuntimeError('prefix cold/warm sequence did not demonstrate reuse')
        prompts = [sha(path.parent / f"{name}-r{repeat}/prompt-1.txt") for repeat in (1, 2, 3)]
        if len(set(prompts)) != 1:
            raise RuntimeError('prefix requests were not exact resends')
    coding_path = root / 'coding-agent-v2/summary.json'
    coding = json.loads(coding_path.read_text())
    if coding['identity'] != live:
        raise RuntimeError('coding and source-review used different images')
    fixture = REPO / 'benchmarks/coding-fixture/v2'
    if coding['fixture_manifest_sha256'] != sha(fixture / 'manifest.json') or coding['tasks_sha256'] != sha(fixture / 'tasks.json'):
        raise RuntimeError('coding fixture differs')
    if coding['metrics']['preemptions'] or not all(row['acceptance']['passed'] for row in coding['task_results']):
        raise RuntimeError('coding acceptance or serving gate failed')
    records = [json.loads(line) for line in (root / 'coding-agent-v2/requests.jsonl').read_text().splitlines()]
    source_finished = datetime.datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo('Europe/Berlin'))
    started = datetime.datetime.fromisoformat(manifest['started_at'])
    return {
        'schema_version': 2, 'date': started.date().isoformat(),
        'release_manifest_sha256': sha(REPO / 'config/production_image.json'),
        'summary_runner_sha256': sha(Path(__file__)),
        'image_id': release['image_id'], 'image_tag': release['image_tag'],
        'policy_sha256': release['policy_sha256'],
        'model_revision': json.loads((REPO / 'config/production_policy.json').read_text())['model']['revision'],
        'power_cap_w': 180,
        'source_review': {
            'run_id': path.parent.name, 'raw_results_path': str(path.relative_to(REPO)),
            'prompt_namespace': manifest['prompt_namespace'],
            'scenario_plan_sha256': sha(REPO / 'benchmarks/current-profile-scenarios.json'),
            'runner_sha256': json.loads((root / 'provenance.json').read_text())['source_sha256']['scripts/current-profile-benchmark.py'],
            'raw_results_sha256': sha(path), 'corpus_sha256': manifest['corpus_sha256'],
            'started_at': manifest['started_at'], 'finished_at': source_finished.isoformat(),
            'finish_time_source': 'final results.json modification time',
            'wall_s': (source_finished - started).total_seconds(),
            'sampling': {**manifest['sampling'], 'thinking': False, 'ignore_eos': True, 'output_tokens': 1024},
            'scenarios': len(groups), 'waves': len(cases),
            'requests': sum(len(row['requests']) for row in cases), 'max_concurrency': 4,
            'logical_prompt_tokens': sum(row['logical_prompt_tokens_observed'] for row in cases),
            'completion_tokens': sum(row['completion_tokens'] for row in cases),
            'preemptions': 0, 'prefill_recompute_excess': 0,
            'all_prompt_counts_match': True, 'all_completion_counts_exact': True, 'all_finish_reasons_length': True,
            'scenario_results': scenario_results, 'prefix_resends': prefix,
            'full_profile_prefix_resends': full_profile_prefix,
            'supplementary_prefix_runs': supplementary,
            'prompt_hash_verification': {'method': 'All measured prompt bytes verified against preflight SHA-256 of the frozen original full-run fixtures, including both prefix scenarios. Request JSON values match the frozen payloads; serialization key order may differ.',
                'matched_requests': sum(len(row['requests']) for row in cases), 'matched_files': len(fixture_hashes), 'mismatched_requests': 0,
                'request_comparison': 'parsed JSON equality; prompt files byte-identical',
                'fixture_manifest_sha256': sha(root / 'fixture-sha256.json'),
                'reference_main_results_path': str(Path(manifest['fixture_root']) / 'results.json')},
        },
        'coding': {**{key: coding[key] for key in ('fixture_id', 'fixture_manifest_sha256', 'fixture_project_sha256', 'tasks_sha256', 'runner_sha256', 'wall_s', 'requests', 'tool_calls', 'metrics')},
            'raw_results_path': str(coding_path.relative_to(REPO)), 'raw_summary_sha256': sha(coding_path),
            'tasks_passed': sum(row['acceptance']['passed'] for row in coding['task_results']),
            'tasks_total': len(coding['task_results']),
            'acceptance_tests_passed': sum(row['acceptance']['tests_run'] for row in coding['task_results']),
            'prompt_tokens_min': min(row['usage']['prompt_tokens'] for row in records),
            'prompt_tokens_max': max(row['usage']['prompt_tokens'] for row in records)},
    }


def readme_measurements(result, public_path):
    source = result['source_review']; coding = result['coding']; metrics = coding['metrics']
    reference = str(public_path.relative_to(REPO))
    rows = {row['name']: row for row in source['scenario_results']}
    if source['supplementary_prefix_runs']:
        date = source['supplementary_prefix_runs'][0]['started_at'][:10]
        prefix_note = (f'Three additional 64K resends on a fresh worker on {date} supply the\n'
                       "cold/warm 64K row, because the complete run's first 64K prefix request\n"
                       'reused 13,312 tokens from its 16K predecessor. Those extra prompts and\n'
                       'payloads match the same frozen fixtures.')
    else:
        prefix_note = 'Both prefix scenarios are cold/warm exact resends within this complete run.'
    text = f'''## Source-review serving benchmark

Measured **{source['started_at'][:10]}** on the permanent service with the
[frozen public corpus](benchmarks/meaningful-corpus.json). The
[current summary]({reference}) records scenario results, fixture hashes,
fixed prompt namespace `20260923-201101` and image identity. Raw prompts,
responses and stream events remain local under `benchmark-results/`.

The complete run covered **{source['scenarios']} scenarios, {source['waves']} measured waves and {source['requests']} successful
requests** in **{source['wall_s']/60:.1f} minutes**. Each request sampled at temperature 1.0,
top-p 0.95 and top-k 20 with thinking disabled. `ignore_eos=true` required
exactly 1,024 output tokens. These are throughput measurements rather than
semantic answer scores. All prompt/output counts matched, all requests
finished at the output cap, with **0 preemptions** and **0 excess recomputed
prefill tokens**. All {source['requests']} measured prompts were verified byte-for-byte
against the frozen original full-run fixtures; request payload values match.
{prefix_note}

### One request: context sweep

Input values are token budgets. Prefill is newly computed KV tokens per native
prefill second; decode is post-first generated tokens per native decode second.
Rates and latencies are medians across waves. MTP acceptance is weighted
across drafted tokens.

| Input / output budget | Waves | Actual input | Prefill tok/s | Decode tok/s | MTP accepted | TTFT | End to end |
|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for name, budget in [('phase-512-c1',512),('phase-2k-c1',2048),('phase-4k-c1',4096),('phase-8k-c1',8192),('phase-16k-c1',16384),('phase-32k-c1',32768),('phase-64k-c1',65536),('phase-128k-c1',131072)]:
        row = rows[name]
        text += f"| {budget:,} / 1,024 | {row['waves']} | {row['actual_prompt_tokens_min']:,}–{row['actual_prompt_tokens_max']:,} | {row['prefill_tps_median']:,.1f} | {row['decode_tps_median']:.1f} | {100*row['mtp_acceptance']:.1f}% | {row['ttft_s_median']:.2f} s | {row['batch_wall_s_median']:.2f} s |\n"
    text += '''
### One to four simultaneous requests

The rates count aggregate generated tokens in sampled intervals after all
requests emitted a first token and before any finished. C1 comes from the
context sweep; C2–C4 each include three waves with different tasks. These are
separate serving load points, not paired scaling measurements.

| Input / output per request | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
'''
    for label, short in [('2,048','2k'),('4,096','4k'),('16,384','16k')]:
        rates = [rows[f'phase-{short}-c1']['fully_overlapped_decode_tps']] + [rows[f'concurrency-{short}-c{c}']['fully_overlapped_decode_tps'] for c in (2,3,4)]
        text += f"| {label} / 1,024 | " + ' | '.join(f'{rate:.1f} tok/s' for rate in rates) + ' |\n'
    if any(row['max_waiting'] for row in source['scenario_results']):
        text += '\nThe scheduler queued some work when capacity was tight; no request was preempted.\n'
    text += '''
### Prefix reuse and maximum context

| Scenario | Measured result |
|---|---|
'''
    for label,name in [('16K','prefix-16k-cold-warm'),('64K','prefix-64k-cold-warm')]:
        prefix = [row for row in source['prefix_resends'] if row['name']==name]
        cold,*warm = prefix
        low, high = min(row['ttft_s'] for row in warm), max(row['ttft_s'] for row in warm)
        warm_time = f'{low:.2f}' if round(low, 2) == round(high, 2) else f'{low:.2f}–{high:.2f}'
        text += f"| {label} exact resend | {int(warm[0]['cached_tokens']):,} / {cold['prompt_tokens']:,} prompt tokens cached; TTFT **{cold['ttft_s']:.2f} s cold → {warm_time} s warm** |\n"
    row=rows['full-context-199680']
    text += f"| Maximum context | **{row['actual_prompt_tokens_min']:,} input + 1,024 output**; {row['prefill_tps_median']:,.1f} prefill tok/s, {row['decode_tps_median']:.1f} decode tok/s, {row['ttft_s_median']:.2f} s TTFT, {row['batch_wall_s_median']:.2f} s end to end |\n"
    text += f'''
The maximum-context row is one capacity and throughput observation. The
16K/64K resends reused the same prompt on the same worker.

## Repeatable coding-agent benchmark

The [QueueKit fixture v2](benchmarks/coding-fixture/v2/README.md) copies a frozen
Python repository and gives the model two linked editing tasks in one
conversation, with file/test tools and hidden acceptance tests after each
task. This fresh run used the same permanent-service image and policy as
the source-review run. The [summary]({reference}) records fixture and runner
hashes. This is one adaptive session, not a multi-run distribution.

| Coding workload result | Measured value |
|---|---:|
| Tasks / hidden acceptance tests | **{coding['tasks_passed']}/{coding['tasks_total']} tasks, {coding['acceptance_tests_passed']}/{coding['acceptance_tests_passed']} tests passed** |
| End-to-end time | **{int(coding['wall_s']//60)} min {coding['wall_s']%60:.0f} s** |
| Model requests / tool calls | **{coding['requests']} / {coding['tool_calls']}** |
| Actual input context range | **{coding['prompt_tokens_min']:,}–{coding['prompt_tokens_max']:,} tokens** |
| Logical prompt / generated tokens | {int(metrics['prompt_tokens']):,} / {int(metrics['generation_tokens']):,} |
| Newly computed / prefix-cached prompt tokens | {int(metrics['prefill_tokens']):,} / {int(metrics['cached_tokens']):,} |
| Prefix-cache hit rate | **{100*metrics['cached_tokens']/metrics['prompt_tokens']:.1f}%** |
| Weighted native prefill compute | **{metrics['prefill_tokens']/metrics['prefill_seconds']:,.1f} tok/s** |
| Weighted native decode | **{metrics['generation_tokens']/metrics['decode_seconds']:.1f} tok/s** |
| MTP accepted / drafted tokens | **{100*metrics['accepted_tokens']/metrics['draft_tokens']:.1f}%** |
| Preemptions | **0** |

The coding rates exclude tool execution; end-to-end time includes it.
Generated tokens include reasoning. Prefix caching remained enabled between
agent turns. Raw generated code and conversation records remain local.

'''
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--update-readme', action='store_true')
    args=parser.parse_args()
    result=summarize(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if args.update_readme:
        path=REPO/'README.md';text=path.read_text()
        start=text.index('## Source-review serving benchmark\n');end=text.index('## Install and serve\n')
        text=text[:start]+readme_measurements(result,args.output.resolve())+text[end:]
        path.write_text(text)
    print(json.dumps({'image':result['image_id'],'scenarios':result['source_review']['scenarios'],
                      'waves':result['source_review']['waves'],'requests':result['source_review']['requests'],
                      'coding_tasks_passed':result['coding']['tasks_passed']},indent=2))


if __name__=='__main__':
    main()
