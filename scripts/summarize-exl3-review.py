#!/usr/bin/env python3
"""Validate and export compact, correctly scoped Pro-review measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.evidence.resolve();receipts={}
    def read(relative):
        path=root/relative;raw=path.read_bytes()
        receipts[relative]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    original=json.loads((REPO/'config/production_image.json').read_text())['image_id']
    first=read('candidate-build-v2/build.json');final=read('candidate-build-c4copy-v3/build.json')
    assert first['status']==final['status']=='PASS'
    assert first['native_manifest']['library_sha256']==final['native_manifest']['library_sha256']
    assert first['native_manifest']['source_sha256']==final['native_manifest']['source_sha256']
    def campaign(name,expected):
        item=read(name+'/campaign.json')
        assert item['status'].startswith('COMPLETE') and item['image_id']==expected,item
        assert item['restoration']['image_id']==original and item['restoration']['models_endpoint_healthy']
        return item
    campaign('focused-xpu-v1',first['image_id'])
    campaign('m04-q4-repeat-v1',first['image_id'])
    campaign('m04-c4copy-final-v1',final['image_id'])
    attention=read('focused-xpu-v1/guarded-attention/result.json')
    assert attention['status'].startswith('PASS') and attention['tolerance']==dict(rtol=.01,atol=.003)
    caches=[]
    for name in ['cache-cap8','cache-default64']:
        d=read('focused-xpu-v1/'+name+'.json')
        assert d['status'].startswith('PASS') and d['library_sha256']==final['native_manifest']['library_sha256']
        assert all(e['final_cache']['peak_entries']<=d['capacity'] for e in d['epochs'])
        assert all(r['allclose'] for r in d['numerical'])
        caches.append(dict(capacity=d['capacity'],exact_lengths=len(d['exact_lengths']),epochs=len(d['epochs']),
            final_cache=d['epochs'][-1]['final_cache'],post_warmup_growth_bytes=d['post_warmup_growth_bytes']))
    copies=[]
    for name in ['focused-xpu-v1','m04-q4-repeat-v1','m04-c4copy-final-v1']:
        d=read(name+'/m04-copy.json');assert d['status'].startswith('PASS')
        assert all(all(c[k] for k in ['bitexact_old_new','native_allclose','fallback_bitexact','graph_mutation_bitexact','output_storage_unchanged']) for c in d['cases'])
        rows=[]
        for c in d['cases']:
            baseline=c['samples_ms']['baseline'];candidate=c['samples_ms']['candidate']
            paired=[100*(statistics.mean(baseline[i:i+2])/statistics.mean(candidate[i:i+2])-1) for i in range(0,len(baseline),2)]
            rng=random.Random(20261002)
            boot=sorted(statistics.mean(rng.choices(paired,k=len(paired))) for _ in range(10000))
            rows.append(dict(batch=c['batch'],rows=c['rows'],kv=c['kv'],median_ms=c['median_ms'],median_speedup_pct=c['speedup_pct'],
                paired_abba_mean_speedup_pct=statistics.mean(paired),paired_round_bootstrap_95_pct=[boot[249],boot[9749]],
                abba_rounds=len(paired)))
        copies.append(dict(campaign=name,numerical_and_mutable_graph_pass=True,cases=rows))
    replay=[]
    for name in ['matched-state-v5','matched-state-full-mixed-v1','matched-state-native-locate-v1','matched-state-early-targets-v1']:
        campaign(name,original);summary=read(name+'/replay/summary.json')
        assert summary['status']=='COMPLETE_MATCHED_STATE_DIAGNOSTIC'
        for path in sorted((root/name/'replay/steps').glob('*-result.json')):
            d=read(str(path.relative_to(root)))
            assert d['status']=='PASS_LOCAL_ATTENTION_ORIGINAL_TOLERANCE'
            assert set(d['routes'])=={'native','native-repeat','m04','onednn','combined'}
            assert all(r['shape'][1]==248320 and r['local_original_tolerance_pass'] and r['initial_state_verified_sha256']==d['restored_state_sha256'] for r in d['routes'].values())
            comparisons={k:{field:value for field,value in c.items() if field!='rows'} for k,c in d['comparisons'].items()}
            selected=[]
            for target in d.get('selected_early_targets',[]):
                row=d['logits_indices'].index(target['target_input_row'])
                selected.append(dict(**target,logit_row=row,comparisons={k:c['rows'][row] for k,c in d['comparisons'].items()}))
            replay.append(dict(campaign=name,case=d['case'],label=d['label'],actual_rows=d['actual_rows'],padded_rows=d['padded_rows'],
                rows_by_request=d['rows_by_request'],graph_mode=d['graph_mode'],
                full_cache_snapshot_bytes=sum(x['bytes'] for x in d.get('full_storage_snapshots',[])),
                addressed_state_saved_bytes=sum(x['saved_bytes'] for x in d['state']),
                comparisons=comparisons,selected_early_targets=selected,
                first_gdn_native_repeat_differences=d.get('first_gdn_native_repeat_differences'),
                original_local_attention_tolerance_pass=True,
                local_comparisons=sum(len(r['local_attention_checks']) for r in d['routes'].values())))
    negatives=read('source-closure-negative.json');assert negatives['status'].startswith('PASS')
    coverage=read('runtime-sources/SOURCE_GUARD_COVERAGE.json')
    assert coverage['required_count']==coverage['covered_count']==13
    result=dict(status='COMPLETE_BOUNDED_REVIEW_TESTS_NOT_RUNTIME_PROMOTION',
        qualified_production_image=original,initial_candidate_image=first['image_id'],final_candidate_image=final['image_id'],
        candidate_native_sha256=final['native_manifest']['library_sha256'],native_reuse_source_and_hash_verified=True,
        source_guard_coverage=dict(required=13,covered=13,missing_and_changed_rejected=True),
        attention_gate=dict(cases=len(attention['cases']),graphs=len(attention['graphs']),causal=len(attention['causal']),tolerance=attention['tolerance']),
        cache_soaks=caches,m04_copy_campaigns=copies,matched_state_replays=replay,
        decision=dict(m04_direct_copy='C4 only; C1-C3 retain the qualified unpack/copy path. Native M04 kernel and split table unchanged.',
            cache='Bounded per-thread/per-queue application cache; exact K preserved. Finite memory plateau; no indefinite memory guarantee.',
            numerical='All local attention comparisons satisfy original thresholds. Real C4 graph/native-repeat is bitidentical. Mixed native-repeat varies even with the whole cache restored; small first b/a projection differences precede GDN and attention. No failed numerical route gate found.',
            limitations='Recreated finite histories/shapes; not an exact reconstruction of all nine historical divergent texts or proof of universally identical logits. Instrumented replay is not throughput. No new BF16, MTP/vocabulary or full 70-wave serving campaign.',
            release='Host helpers/docs can be integrated without a model-image change. Candidate runtime requires a distinct image/policy attestation and relevant serving/restart qualification before production promotion.'),
        local_receipt_sha256=receipts)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],candidate_image=final['image_id'],matched_replays=len(replay),local_attention_comparisons=sum(x['local_comparisons'] for x in replay)),indent=2))


if __name__=='__main__':main()
