"""Publish compact receipts, preserving raw files and their original statuses."""
import hashlib
import json
from pathlib import Path
import statistics

REPO=Path(__file__).resolve().parents[3]
RAW=REPO/'benchmark-results/dflash2-serving-v031-20261009'
OUT=REPO/'benchmarks/results/dflash2-pro-optimization-v031'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())


def publish():
    OUT.mkdir(exist_ok=True)
    rows=[]
    for name in ['pro-probabilistic-k7','pro-w4-k7-attempt3','pro-w4-group4-k7','pro-w4-group4-kv-only-k7','pro-w4-group5-matched96','pro-w4-group4-matched96']:
        folder=RAW/name
        if not (folder/'campaign.json').exists(): continue
        c=read(folder/'campaign.json')
        if 'finished_unix' not in c or 'benchmark_results' not in c: continue
        assert c['status'].startswith('PASS_'),c['status']
        assert c['restoration']['image_id']==c['production_image']
        assert c['restoration']['models_endpoint_healthy']
        path=REPO/c['benchmark_results'];assert digest(path)==c['benchmark_sha256']
        b=read(path)
        row=dict(name=name,image=c['image'],raw_status=c['status'],
            draft_format=c.get('draft_format','fp16' if name=='pro-probabilistic-k7' else 'w4'),
            draft_sample_method=c['draft_sample_method'],max_model_len=c['active_context_limit'],
            raw_results=c['benchmark_results'],raw_sha256=digest(path),rollback=c['restoration'],
            scenario_results=[])
        if name=='pro-w4-k7-attempt3':
            row['legacy_status_note']='The original controller emitted an unquantized label for this W4 arm. Actual image identity, W4 role inventory and packed GPU probes establish its format; the raw campaign was not rewritten.'
        for scenario in dict.fromkeys(x['scenario']['name'] for x in b['cases']):
            cases=[x for x in b['cases'] if x['scenario']['name']==scenario]
            assert len(cases)==3
            assert all(x['completion_tokens']==1024 and x['preemptions']==0 for x in cases)
            rates=[x['native_weighted_decode_tokens_per_s'] for x in cases]
            row['scenario_results'].append(dict(name=scenario,decode_tps=rates,
                median_decode_tps=statistics.median(rates),
                preemptions=[x['preemptions'] for x in cases],
                acceptance=[x['speculative_acceptance'] for x in cases],
                output_tokens=[x['completion_tokens'] for x in cases]))
        receipt=OUT/name;receipt.mkdir(exist_ok=True)
        case_fields=['repeat','native_weighted_decode_tokens_per_s','native_decode_seconds','post_first_completion_tokens',
            'completion_tokens','requested_prompt_tokens','logical_prompt_tokens_observed',
            'prefill_tokens_computed','prompt_tokens_cached','prefill_recompute_excess',
            'native_prefill_seconds','native_prefill_compute_tokens_per_s','preemptions',
            'speculative_draft_tokens','speculative_accepted_tokens','speculative_acceptance',
            'batch_wall_s','request_ttft_s','request_e2e_s','finished_requests_delta']
        cases=[]
        for x in b['cases']:
            assert x['native_decode_seconds']>0
            expected=x['post_first_completion_tokens']/x['native_decode_seconds']
            assert abs(expected-x['native_weighted_decode_tokens_per_s'])<1e-9
            value={k:x[k] for k in case_fields};value['scenario']=x['scenario']['name']
            value['responses']=[{k:r[k] for k in ['usage','finish_reason','output_sha256','wall_s','ttft_s']} for r in x['requests']]
            cases.append(value)
        (receipt/'native-cases.json').write_text(json.dumps(cases,indent=2)+'\n')
        manifest=[]
        for file in ['campaign.json','worker-attestation.json','w4-numerical.json','prefix-state.json','exact-context.json','context-end-state.json','cache-ledger-rpc.json','greedy-control.json','proposal-audit.json']:
            p=folder/file
            if not p.exists():continue
            manifest.append(dict(path=str(p.relative_to(REPO)),sha256=digest(p),bytes=p.stat().st_size))
            value=read(p)
            if file=='campaign.json':
                # Keep identities and rollback; bulky checkpoint conversion
                # inventory already lives with the converted checkpoint.
                value={k:v for k,v in value.items() if k not in ['draft','benchmark_command']}
                value['draft_checkpoint_output_sha256']=c['draft'].get('output_sha256',c['draft'].get('fp16_sha256'))
            if file=='exact-context.json':value={k:v for k,v in value.items() if k!='response'}
            if file=='context-end-state.json':
                value=dict(status=value['status'],preemptions=value['preemptions'],scope=value['scope'],
                    sampling_usage=value['sampling']['usage'],
                    sampling_finish=value['sampling']['choices'][0]['finish_reason'],
                    custom_stop_usage=value['custom_stop']['usage'],
                    custom_stop_finish=value['custom_stop']['choices'][0]['finish_reason'])
            (receipt/file).write_text(json.dumps(value,indent=2)+'\n')
        ledger=folder/f"context-{c['active_context_limit']}"/'cache-ledger.json'
        if ledger.exists():
            value=read(ledger)
            manifest.append(dict(path=str(ledger.relative_to(REPO)),sha256=digest(ledger),bytes=ledger.stat().st_size))
            source_manifest=value.pop('sources')
            (receipt/'cache-ledger.json').write_text(json.dumps(value,indent=2)+'\n')
            (receipt/'runtime-source-manifest.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
        (receipt/'raw-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        rows.append(row)
    result=dict(status='DIAGNOSTIC_SCREENS_NOT_PRODUCTION_QUALIFIED',rows=rows,
        scope='All three repeats retained. Initial grouping screen changes grouping and configured max context (96K to192K); it is not an isolated speed attribution. Stochastic outputs differ and require independent law/state gates. Historical figures are references, not paired promotion evidence.')
    (OUT/'initial-screens.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PUBLISHED_COMPLETED_SCREENS',len(rows))


if __name__=='__main__': publish()
