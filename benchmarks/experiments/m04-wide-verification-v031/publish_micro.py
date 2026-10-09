#!/usr/bin/env python3
"""Append exact kernel and cache evidence to the serving receipt manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'benchmark-results/m04-wide-verification-v031-20261009'
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
manifest_path = args.out / 'MANIFEST.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['production_promoted'] is False

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

files = {
    'kernel-micro.json': ROOT / 'micro-adaptive/probe/result.json',
    'kernel-campaign.json': ROOT / 'micro-adaptive/campaign.json',
    'block9-fp16.json': ROOT / 'micro-adaptive/block9.json',
    'kernel-build.json': ROOT / 'build-adaptive/output/manifest.json',
    'kernel-build-controller.json': ROOT / 'build-adaptive/controller.json',
    'cache-regression.json': ROOT / 'cache-regression.json',
    'cache-original-failed-campaign.json': REPO / 'benchmark-results/dflash2-serving-v031-20261009/wide-m04-k8-attempt2/campaign.json',
}
micro = json.loads(files['kernel-campaign.json'].read_text())
assert micro['status'] == 'PASS_NEW_M04_MICRO_GATES_NOT_SERVING_QUALIFICATION'
assert micro['restoration']['models_endpoint_healthy']
assert micro['restoration']['image_id'] == micro['production_image']
assert sha(files['kernel-micro.json']) == micro['result_sha256']
assert json.loads(files['block9-fp16.json'].read_text())['status'] == 'PASS_NEW_BLOCK9_FP16_PRIMITIVES'
assert json.loads(files['cache-regression.json'].read_text())['status'] == 'PASS_SPECULATIVE_DEPTH_CACHE_ISOLATION'
for name, source in files.items():
    assert not (args.out / name).exists()
    shutil.copyfile(source, args.out / name)
    manifest['sources'][str(source.relative_to(REPO))] = sha(source)
log = REPO / 'benchmark-results/dflash2-serving-v031-20261009/wide-m04-k8-attempt2/context-96000/worker.log'
lines = log.read_text().splitlines()
errors = [line for line in lines if 'expected size 8==7' in line or '40960==35840' in line]
assert errors, 'preserve the observed cache-shape failure'
(args.out / 'cache-failure.json').write_text(json.dumps(dict(
    status='REPRODUCED_DRAFT_LENGTH_AOT_CACHE_COLLISION_BEFORE_PATCH',
    affected_installed_version='vLLM 0.31.0', original_log_sha256=sha(log),
    evidence=errors, patch='SpeculativeConfig.compute_hash includes num_speculative_tokens',
    numeric_kernel_failure=False), indent=2) + '\n')
manifest['sources'][str(log.relative_to(REPO))] = sha(log)
dispatch = []
for receipt in sorted(args.out.glob('campaign-*.json')):
    campaign = json.loads(receipt.read_text())
    if 'benchmark_results' not in campaign:
        continue
    folder = (REPO / campaign['benchmark_results']).parents[2]
    log = folder / f"context-{campaign['active_context_limit']}/worker.log"
    text = log.read_text()
    rows = campaign['depth'] + 1
    routes = sorted(set(line for line in text.splitlines() if 'EXL3_GUARDED_ATTN_ROUTE' in line))
    assert any(f"('M04', 1, {rows}, True)" in line for line in routes)
    kv_gib = re.findall(r'Available KV cache memory: ([\d.]+) GiB', text)
    kv_tokens = re.findall(r'XPU KV cache size: ([\d,]+) tokens', text)
    assert kv_gib and kv_tokens
    dispatch.append(dict(campaign=receipt.name, method=campaign.get('method', 'dflash'),
                         depth=campaign['depth'], image=campaign['image'],
                         profiled_kv_gib=float(kv_gib[-1]),
                         reported_cache_token_capacity=int(kv_tokens[-1].replace(',', '')),
                         raw_log_sha256=sha(log), routes=routes,
                         aot_artifact_keys=sorted(set(re.findall(r'torch_aot_compile/([a-f0-9]{64})/', text)))))
    manifest['sources'][str(log.relative_to(REPO))] = sha(log)
fixed = [row for row in dispatch if row['image'].startswith('sha256:b0b9244420a2')]
assert {('dflash', 4), ('dflash', 8), ('mtp', 4)} <= {(row['method'], row['depth']) for row in fixed}
for index, left in enumerate(fixed):
    assert left['aot_artifact_keys']
    for right in fixed[index + 1:]:
        assert set(left['aot_artifact_keys']).isdisjoint(right['aot_artifact_keys'])
(args.out / 'serving-dispatch-cache.json').write_text(json.dumps(dict(
    status='PASS_ACTUAL_M04_DISPATCH_AND_DEPTH_ISOLATED_AOT_ARTIFACTS', arms=dispatch), indent=2) + '\n')
draft_path = Path.home() / '.cache/exl3xpu/incoai-Qwen3.8-27B-DFlash2-dedf8df6-fp16/config.json'
draft = json.loads(draft_path.read_text())
assert sha(draft_path) == '45b073285ccec54c62ffafdb726d2676d4965dc4fe9ef2256cc3a6933e61b8f3'
assert draft['num_hidden_layers'] == 5 and draft['sliding_window'] == 2048
target_path = Path.home() / '.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw/config.json'
target = json.loads(target_path.read_text())
text_config = target.get('text_config', target)
mtp_source = REPO / 'benchmark-results/vllm-031-upgrade-20261009/installed-vllm/model_executor/models/qwen3_5_mtp.py'
assert 'layer_type="full_attention"' in mtp_source.read_text()
assert text_config.get('mtp_num_hidden_layers', 1) == 1
(args.out / 'cache-structure.json').write_text(json.dumps(dict(
    status='PINNED_MODEL_CACHE_STRUCTURE_NOT_ALLOCATED_MEMORY_MEASUREMENT',
    target_config_sha256=sha(target_path), draft_config_sha256=sha(draft_path),
    mtp_source_sha256=sha(mtp_source), mtp_own_full_attention_layers=1,
    dflash_own_sliding_attention_layers=draft['num_hidden_layers'],
    dflash_window_tokens=draft['sliding_window'],
    dflash_kv_heads=draft['num_key_value_heads'], dflash_head_dim=draft['head_dim'],
    target_kv_replaced_or_duplicated=False,
    note='Verification uses the ordinary target cache in both methods. Draft layers hold their own projected K/V; shared allocation is not shared layer contents. Grouping, prefix caching and speculative reservations can exceed the active window footprint.'), indent=2) + '\n')
for source in Path(__file__).parent.rglob('*'):
    if source.is_file() and 'build' not in source.relative_to(Path(__file__).parent).parts and '__pycache__' not in str(source):
        manifest['sources'][str(source.relative_to(REPO))] = sha(source)
manifest['public_files'] = {p.name: sha(p) for p in args.out.iterdir() if p.is_file() and p.name != 'MANIFEST.json'}
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print(args.out)
