#!/usr/bin/env python3
"""Check public docs against release receipts; --write refreshes marked tables.

CPU-only, standard library, no network/Docker/service calls. Archived reports
are checked for local link targets but are not rewritten as current results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from release_integrity import load_release

REPO = Path(__file__).resolve().parents[1]
DOCS = [
    'README.md', 'docs/INSTALLATION.md', 'docs/CURRENT_CONFIGURATION.md',
    'docs/OPERATIONS.md', 'docs/BENCHMARKS.md', 'docs/ARCHIVE.md',
    'docs/EXL3_BENCHMARK_REPRODUCTION.md',
    'docs/EXL3_PRODUCTION_MIGRATION_AND_OPTIMIZATION_PLAN.md',
    'docs/archive/README-before-consolidation-20261114.md',
]


def read_json(path):
    return json.loads((REPO / path).read_text())


def table(rows):
    return '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)


def receipt(record):
    path = REPO / record['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Changed evidence receipt: ' + record['path'])
    return json.loads(path.read_text())


def blocks():
    release = load_release(REPO)
    policy = read_json('config/production_policy.json')
    ram = read_json('config/production_ram_kv.json')
    decision = receipt(release['qualification_review'])
    summary = receipt(decision['benchmark_summary'])
    if summary['release_image'] != release['image_id']:
        raise ValueError('Benchmark summary is for another release')
    # Deliberately select a dated campaign. A future release must review this
    # source and the dated prose, rather than silently inherit old numbers.
    campaign = 'benchmarks/results/mtp-cache-1792-20261112/qualification/full-serving-20261112-r2/comparison.json'
    comparison = read_json(campaign)
    if (comparison['changes_pct'] != summary['changes_pct'] or
            not comparison['all70_waves_valid_each'] or
            not comparison['paired_payloads_and_prompt_bytes_verified']):
        raise ValueError('Serving comparison and qualified summary differ')

    rollback_id = policy['release']['immediate_exl3_rollback_image']
    saved = [p for p in (REPO / 'config/releases').glob('*/production_image.json')
             if json.loads(p.read_text())['image_id'] == rollback_id]
    if len(saved) != 1:
        raise ValueError('Expected one saved immediate rollback')
    rollback = saved[0].parent.name
    launcher = f'scripts/run-server-{rollback}-rollback.sh'
    if not (REPO / launcher).is_file():
        raise ValueError('Missing rollback launcher: ' + launcher)

    profile = table([
        ['Setting', 'Supported value'], ['---', '---'],
        ['GPU / power cap', f"{policy['hardware']['gpu_count']} × {policy['hardware']['gpu']} (32 GB) / {policy['hardware']['power_limit_w']} W"],
        ['Model', policy['model']['served_name'] + ' / EXL3 4.00 bpw'],
        ['Runtime', f"vLLM {policy['runtime']['vllm']}, Torch {policy['runtime']['torch']}"],
        ['Context budget', f"{policy['serving']['max_model_len']:,} input + output tokens"],
        ['Speculative decoding', f"MTP, {policy['speculation']['speculative_tokens']} draft tokens"],
        ['Prefix cache', f"FP8 GPU KV + {ram['default_cpu_gib']} GiB host RAM"],
        ['API', '`http://127.0.0.1:8081/v1`, model `' + policy['model']['served_name'] + '`'],
    ])
    pins = table([
        ['Identity', 'Pinned value'], ['---', '---'],
        ['Image tag', f"`{release['image_tag']}`"],
        ['Image ID', f"`{release['image_id']}`"],
        ['Policy ID', f"`{policy['policy_id']}`"],
        ['Policy SHA256', f"`{release['policy_sha256']}`"],
        ['Checkpoint', f"`{policy['model']['id']}`"],
        ['Checkpoint revision', f"`{policy['model']['revision']}`"],
        ['vLLM / Torch', f"{policy['runtime']['vllm']} / {policy['runtime']['torch']}"],
        ['RAM layout / default budget', f"{ram['kv_cache_layout']} / {ram['default_cpu_gib']} GiB"],
    ])
    scenarios = {row['scenario']: row for row in comparison['scenarios']}
    measurements = [
        ['Workload', 'Waves', 'First token', 'Decode tok/s', 'Wave time'],
        ['---', '---:', '---:', '---:', '---:'],
    ]
    for name, label in [
        ('phase-4k-c1', '4K cold, one request'),
        ('phase-64k-c1', '64K cold, one request'),
        ('phase-128k-c1', '128K cold, one request'),
        ('prefix-16k-cold-warm', '16K prefix reuse, one request'),
        ('prefix-64k-cold-warm', '64K prefix reuse, one request'),
        ('concurrency-4k-c4', '4K each, four requests (combined decode)'),
    ]:
        row = scenarios[name]
        metric = row['metrics']
        decode_key = ('fully_overlapped_aggregate_decode_tokens_per_s'
                      if row['concurrency'] > 1 else 'native_weighted_decode_tokens_per_s')
        measurements.append([
            label, str(row['waves']),
            f"{metric['request_ttft_median_s']['candidate_median']:.2f} s",
            f"{metric[decode_key]['candidate_median']:.2f}",
            f"{metric['batch_wall_s']['candidate_median']:.2f} s",
        ])
    rollback_text = (f'Use the [saved {rollback} release](../config/releases/{rollback}/README.md)\n'
                     f'with `{launcher}`. Its immutable image is\n`{rollback_id}`.')
    return {
        ('README.md', 'PUBLIC PROFILE'): profile,
        ('README.md', 'PUBLIC MEASUREMENTS'): table(measurements),
        ('docs/CURRENT_CONFIGURATION.md', 'RELEASE IDENTITIES'): pins,
        ('docs/OPERATIONS.md', 'IMMEDIATE ROLLBACK'): rollback_text,
    }


def check_links():
    errors = []
    for name in DOCS:
        path = REPO / name
        # Markdown links in fenced examples are not navigation targets.
        source = re.sub(r'^```.*?^```[^\n]*', '', path.read_text(), flags=re.M | re.S)
        for target in re.findall(r'\]\(([^\s)]+)\)', source):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            dest = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if not dest.exists():
                errors.append(f'{name}: missing link target {target}')
            elif parsed.fragment and dest.suffix == '.md':
                headings = []
                for heading in re.findall(r'^#{1,6}\s+(.+)$', dest.read_text(), re.M):
                    slug = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
                    headings.append(slug)
                if unquote(parsed.fragment) not in headings:
                    errors.append(f'{name}: missing heading {target}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    errors = []
    for (name, label), expected in blocks().items():
        path = REPO / name
        text = path.read_text()
        start, end = f'<!-- BEGIN {label} -->', f'<!-- END {label} -->'
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError('Expected unique markers: ' + name + ' / ' + label)
        before, rest = text.split(start)
        actual, after = rest.split(end)
        if actual != '\n' + expected + '\n':
            if args.write:
                path.write_text(before + start + '\n' + expected + '\n' + end + after)
            else:
                errors.append(f'{name}: stale {label} table; run with --write after reviewing evidence')
    errors.extend(check_links())
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Public docs: release receipts, profile, benchmark/rollback tables and local links pass.')


if __name__ == '__main__':
    main()
