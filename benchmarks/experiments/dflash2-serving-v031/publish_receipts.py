#!/usr/bin/env python3
"""Write local public experiment receipts; never deploy or push a branch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, action='append', required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'preserve previous public receipts'
    comparison = json.loads(args.comparison.read_text())
    assert comparison['status'] == 'EXPERIMENT_ONLY_NOT_PRODUCTION_QUALIFICATION'
    assert len(comparison['arms']) == len(args.campaign)
    args.out.mkdir(parents=True)
    shutil.copyfile(args.comparison, args.out / 'comparison.json')
    sources = {}
    for path in args.campaign:
        path = path.resolve()
        campaign = json.loads(path.read_text())
        method = campaign.get('method', 'dflash')
        assert campaign['status'] in ['PASS_UNQUANTIZED_DFLASH_STARTUP_ACCEPTANCE_AND_SPEED', 'PASS_CHECKPOINT_NATIVE_MTP_STARTUP_ACCEPTANCE_AND_SPEED']
        assert campaign['restoration']['models_endpoint_healthy']
        assert campaign['restoration']['image_id'] == campaign['production_image']
        assert (campaign['active_context_limit'] == 96000 if method == 'dflash'
                else 192000 <= campaign['active_context_limit'] <= 262144)
        depth = campaign['depth']
        label = f'k{depth}' if method == 'dflash' else f'mtp-k{depth}'
        matching = [arm for arm in comparison['arms'] if arm['depth'] == depth and arm['benchmark_sha256'] == campaign['benchmark_sha256']]
        assert len(matching) == 1
        assert matching[0]['benchmark_sha256'] == campaign['benchmark_sha256']
        assert len(matching[0]['matching_frozen_request_sha256']) == 6 * len(matching[0]['measurements'])
        raw = REPO / campaign['benchmark_results']
        assert sha(raw) == campaign['benchmark_sha256']
        shutil.copyfile(raw, args.out / f'raw-{label}.json')
        shutil.copyfile(path, args.out / f'campaign-{label}.json')
        if path.name == 'campaign-reviewed.json':
            shutil.copyfile(path.parent / 'campaign.json', args.out / f'controller-k{depth}-original-failed.json')
        candidates = [path.parent / 'executing-controller.py', *path.parent.parent.glob('running-controller*.py')]
        executing = [p for p in candidates if p.exists()
                     if sha(p) == campaign['sources']['run_serving.py']]
        assert len(executing) == 1
        shutil.copyfile(executing[0], args.out / f'{label}-executing-controller.txt')
        for name in ['worker-attestation.json', 'prefix-state.json', 'smoke.json']:
            shutil.copyfile(path.parent / name, args.out / f'{label}-{name}')
        audits = list((path.parent / f"context-{campaign['active_context_limit']}/loader").glob('loader-*.json'))
        assert len(audits) == 1
        audit = json.loads(audits[0].read_text())
        mtp_modules = 8 if method == 'mtp' else 0
        assert audit['status'] == 'PASS' and audit['loaded_modules'] == audit['expected_modules'] == 401 + mtp_modules
        assert audit['mtp_loaded'] == audit['mtp_expected'] == mtp_modules
        public_audit = {k: v for k, v in audit.items() if k != 'layers'}
        public_audit.update(original_sha256=sha(audits[0]), note='Per-layer inventory retained in raw local evidence; strict target audit summary.')
        write(args.out / f'{label}-loader-audit.json', public_audit)
        exact = path.parent / 'exact-context.json'
        if exact.exists():
            receipt = json.loads(exact.read_text())
            assert receipt['status'] == 'PASS_EXACT_CONTEXT_CAPACITY_NOT_SEMANTIC_QUALITY'
            assert receipt['usage']['total_tokens'] == 96000
            shutil.copyfile(exact, args.out / f'{label}-exact-context.json')
        sources[str(path.relative_to(REPO))] = sha(path)
        sources[str(raw.relative_to(REPO))] = sha(raw)
        sources[str(audits[0].relative_to(REPO))] = sha(audits[0])
    for file in Path(__file__).parent.iterdir():
        if file.is_file():
            sources[str(file.relative_to(REPO))] = sha(file)
    manifest = dict(status='EXPERIMENT_ONLY_NOT_PRODUCTION_QUALIFICATION', sources=sources,
                    measured_scenario_names=sorted({m['scenario'] for arm in comparison['arms'] for m in arm['measurements']}),
                    source_history_note='Source hashes record versions at publication. Executing controller snapshots preserve the measured campaigns; later experiment revisions may differ.',
                    baseline_regenerated=False, quantized_draft_created=False,
                    eventual_serving_minimum_tokens=192000, temporary_unquantized_limit_tokens=96000,
                    production_promoted=False, historical_power_comparisons_repeated=False,
                    public_files={file.name: sha(file) for file in args.out.iterdir() if file.is_file()})
    write(args.out / 'MANIFEST.json', manifest)
    print(args.out)

if __name__ == '__main__':
    main()
