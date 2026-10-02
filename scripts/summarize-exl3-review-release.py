#!/usr/bin/env python3
"""Export passed exact-image follow-up gates without approving a failed check."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serving',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--release-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();q=a.qualification.resolve();root=a.out.resolve();assert not root.exists()
    sources={}
    def read(path):
        path=Path(path);sources[str(path.relative_to(REPO))]=sha(path);return json.loads(path.read_text())
    campaign=read(q/'campaign.json');serving=read(a.serving/'campaign.json')
    assert campaign['status']=='PASS_BOUNDED_REVIEW_RELEASE_QUALIFICATION'
    assert set(campaign['phases'])=={'operations','matched','quality','telemetry'}
    assert all(x['status']=='PASS' for x in campaign['phases'].values())
    assert serving['status']=='COMPLETE_REVIEW_SERVING_MATRIX'
    image=campaign['image_id'];release=read(a.release_dir/'production_image.json')
    assert release['image_id']==image==serving['image_receipt']['image_id']
    assert sha(a.release_dir/'production_policy.json')==campaign['policy_sha256']==release['policy_sha256']
    operations=read(q/'operations/operations.json');matched=read(q/'matched/assessment.json')
    quality=read(q/'quality/quality-smoke.json');cache=read(q/'telemetry/assessment.json')
    references={}
    for label,base,used_windows in [('short','quantization-reference-20261001',16),('long','exl3-long-quality-v3',1)]:
        original=REPO/'benchmark-results'/base
        reference=read(original/'bf16/summary.json')
        assert reference['panel_sha256']==sha(original/'panel.json')==sha(q/'quality'/label/'panel.json')
        assert reference['vocab_size']==248320
        arrays={}
        for index in range(used_windows):
            for kind in ('nll','logprobs'):
                path=original/'bf16'/f'window-{index:03d}-{kind}.npy'
                arrays[str(path.relative_to(REPO))]=sha(path)
        references[label]=dict(panel_sha256=reference['panel_sha256'],reference_summary_sha256=sha(original/'bf16/summary.json'),
            reference_arrays_sha256=arrays,reused_windows=used_windows,regenerated=False)
    assert operations['status']=='PASS_EXACT_IMAGE_OPERATIONAL_GATES'
    assert matched['status']=='PASS_MATCHED_SERVING_SCREEN' and not any(x['trigger'] for x in matched['repeat'])
    assert quality['status']=='PASS_COMPACT_REFERENCE_SMOKE' and all(x['bitidentical'] for x in quality['short_v1_arrays'])
    assert cache['status']=='PASS_ACTUAL_WORKER_CACHE_DIAGNOSTIC'
    boundary=operations['cases']['near-limit'];assert boundary['usage']['prompt_tokens']==261120 and boundary['usage']['completion_tokens']==1024 and boundary['native']['preemptions']==0
    assert operations['cases']['c4-long']['native']['preemptions']==0
    assert operations['confirmed_abort_trace_occurrences']>0 and operations['cases']['worker-restart']['status']=='PASS'
    assert operations['loader']['loaded_modules']==409 and operations['loader']['mtp_loaded']==8
    coverage=read(q/'operations/runtime-sources/SOURCE_GUARD_COVERAGE.json')
    assert coverage['required_count']==coverage['covered_count']==13
    captured=read(q/'operations/runtime-sources/CAPTURE_IDENTITY.json');assert captured['image_id']==image
    for name,digest in coverage['files_sha256'].items():assert sha(q/'operations/runtime-sources'/name)==digest
    replays=[]
    for path in sorted((q/'quality/replay/observations/steps').glob('*-result.json')):
        d=read(path);assert d['status']=='PASS_LOCAL_ATTENTION_ORIGINAL_TOLERANCE'
        assert all(r['local_original_tolerance_pass'] and r['shape'][1]==248320 and r['initial_state_verified_sha256']==d['restored_state_sha256'] for r in d['routes'].values())
        replays.append(dict(label=d['label'],graph_mode=d['graph_mode'],rows_by_request=d['rows_by_request'],
            local_attention_comparisons=sum(len(x['local_attention_checks']) for x in d['routes'].values()),
            comparisons={k:{field:value for field,value in x.items() if field!='rows'} for k,x in d['comparisons'].items()}))
    assert len(replays)==4 and any(x['graph_mode']=='FULL' for x in replays) and any(x['label']=='long-prefill' for x in replays)
    s=importlib.util.spec_from_file_location('serving_export',REPO/'scripts/summarize-readme-benchmarks.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    summary=m.summarize(a.serving.resolve(),serving_only=True)
    assert summary['image_id']==image and summary['source_review']['waves']==70 and summary['source_review']['requests']==124
    root.mkdir(parents=True)
    (root/'serving-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    quality.pop('short_v1_arrays');quality['short_v1_bitidentical_aggregate_arrays']=32
    assessment=dict(status='PASS_EXACT_IMAGE_RELEASE_GATES',image_id=image,policy_sha256=release['policy_sha256'],
        runtime_parent_image_id=release['runtime_image_id'],source_commit=release['source_commit'],
        runtime_artifacts_sha256=release['runtime_artifacts_sha256'],metadata_attestation=release['rootfs_metadata_attestation'],
        serving=dict(scenarios=20,waves=70,requests=124,preemptions=0,summary_sha256=sha(root/'serving-summary.json')),
        operations=operations,matched_comparison=matched,quality_smoke=quality,reference_provenance=references,candidate_state_replays=replays,
        actual_worker_cache=cache,source_guard_coverage=dict(required=13,covered=13,files=len(coverage['files_sha256'])),
        rollback_image_id=campaign['baseline_image_id'],receipts_sha256=sources,
        limitations=['Finite cache workload; application cache cap is per thread/queue and does not bound oneDNN internal caches or every allocator.',
            'Client p95 SSE burst gaps are not GPU token-step latency.',
            'Historical Flappy and QueueKit were not rerun or relabeled as candidate results; no general coding-quality ranking.',
            'C16 is pressure with permitted preemptions; clean C4 and one maximum context are separate operating claims.',
            'Quality smoke reuses existing BF16 references; short arrays alone do not prove long attention/verification correctness. Fresh graph/long-prefill numerical replay is separate.'])
    (root/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    decision=dict(status='QUALIFIED_FOR_PRODUCTION_PROMOTION',image_id=image,policy_sha256=release['policy_sha256'],
        source_commit=release['source_commit'],native_library_sha256=release['runtime_artifacts_sha256']['/opt/exl3xpu/exl3xpu/_C.so'],
        gate_assessment=dict(path=str((root/'assessment.json').relative_to(REPO)),sha256=sha(root/'assessment.json')),
        benchmark_summary=dict(path=str((root/'serving-summary.json').relative_to(REPO)),sha256=sha(root/'serving-summary.json')),
        immediate_rollback_image_id=campaign['baseline_image_id'],
        rationale='Exact-image serving, original-tolerance numerical, API/media, capacity, abort/restart and bounded actual-worker cache gates passed; matched serving screen has no unresolved repeated review-trigger regression. Safety/resource bound justifies equivalent performance without claiming a universal speedup.',
        deployment='Pending strict-launcher switch and post-deployment API/image checks; this decision does not itself start a service.')
    (root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(dict(status=decision['status'],image_id=image,out=str(root)),indent=2))


if __name__=='__main__':main()
