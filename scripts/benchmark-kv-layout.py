#!/usr/bin/env python3
"""Cold-cache depth checks using Sergio's pinned, unchanged SSE timing function.

This wrapper imports the external MIT-licensed harness; it does not change the
server or power cap. See benchmarks/results/ram-kv-layout-20261108/README.md.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request
import uuid

HARNESS_SHA256 = '45d16c62eb2c14cb5bc7e2e8d586b7c077bb93b8e016dca371c7bdfa27224f33'
PROMPTS_SHA256 = '74afe579763ef91a9321bfc9ab43ea0f55959176c7423347ab5872dbd73eca64'


def metrics(base):
    with urllib.request.urlopen(base + '/metrics', timeout=10) as response:
        raw = response.read().decode()
    counters = {}
    for line in raw.splitlines():
        if line.startswith('vllm:'):
            name = line.split('{')[0].split()[0]
            counters[name] = counters.get(name, 0) + float(line.split()[-1])
    return counters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-dir', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--base', default='http://127.0.0.1:8081')
    p.add_argument('--model', default='Qwen3.8-27B')
    p.add_argument('--hwmon', required=True, type=Path)
    p.add_argument('--power-w', type=int, default=180)
    p.add_argument('--depths', default='32k,128k')
    p.add_argument('--kinds', default='code')
    p.add_argument('--reps', type=int, default=2)
    p.add_argument('--sampled', action='store_true')
    a = p.parse_args()
    harness, prompts_file = [a.upstream_dir / f for f in ('depth_bench.py', 'prompts-qwen38.json')]
    for path, expected in ((harness, HARNESS_SHA256), (prompts_file, PROMPTS_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen upstream file changed: ' + path.name)
    if a.reps < 1:
        p.error('--reps must be positive')
    a.out.mkdir(parents=True, exist_ok=True)
    result_path = a.out / 'cells.jsonl'
    # Refuse to mix server generations or accidentally append duplicate cells.
    if result_path.exists():
        raise RuntimeError('Choose a fresh output directory')
    spec = importlib.util.spec_from_file_location('sergio_depth_bench', harness)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    prompts = json.loads(prompts_file.read_text())
    args = SimpleNamespace(base=a.base, model=a.model, max_tokens=512,
                           timeout=900, hwmon=str(a.hwmon))

    def check_power():
        if int((a.hwmon / 'power1_cap').read_text()) != a.power_w * 1_000_000:
            raise RuntimeError('Unexpected power cap; this script never changes it')

    check_power()
    extra = {'reasoning_effort': 'none', 'cache_salt': uuid.uuid4().hex}
    if a.sampled:
        extra.update(temperature=1.0, top_p=0.95, top_k=20)
    bench.ask(args, 'Say ok.', extra)
    for rep in range(a.reps):
        for depth in a.depths.split(','):
            for kind in a.kinds.split(','):
                prompt = prompts[f'{kind}-{depth}']
                assert hashlib.sha256(prompt['content'].encode()).hexdigest() == prompt['sha256']
                check_power()
                before = metrics(a.base)
                assert before['vllm:num_requests_running'] == before['vllm:num_requests_waiting'] == 0
                extra['cache_salt'] = uuid.uuid4().hex
                row, _ = bench.ask(args, prompt['content'], extra)
                after = metrics(a.base)
                delta = {k: value - before.get(k, 0) for k, value in after.items()
                         if k.endswith(('_total', '_count', '_sum'))}
                checks = {
                    '512_tokens': row['completion_tokens'] == 512,
                    'no_gpu_hits': delta.get('vllm:prefix_cache_hits_total') == 0,
                    'no_cpu_hits': delta.get('vllm:external_prefix_cache_hits_total') == 0,
                    'no_preemptions': delta.get('vllm:num_preemptions_total') == 0,
                    'one_request': delta.get('vllm:request_success_total') == 1,
                    'no_foreign_output': delta.get('vllm:generation_tokens_total') == 512,
                    'no_reasoning': row['last_chunk_extra']['usage']['completion_tokens_details']['reasoning_tokens'] == 0,
                    'idle_after': after['vllm:num_requests_running'] == after['vllm:num_requests_waiting'] == 0,
                }
                row.update(rep=rep, depth=depth, kind=kind, power_cap_w=a.power_w,
                           mode='sampled' if a.sampled else 'greedy', prompt_sha256=prompt['sha256'],
                           metrics_delta=delta, checks=checks, valid=all(checks.values()),
                           utc=dt.datetime.now(dt.UTC).isoformat())
                with result_path.open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(json.dumps({k: row[k] for k in ('rep', 'depth', 'kind', 'ttft_s', 'decode_tok_s', 'valid')}), flush=True)
                check_power()
                if not row['valid']:
                    raise RuntimeError('Cell failed isolation/cache/token checks; see saved result')


if __name__ == '__main__':
    main()
