"""Keep deterministic controls and stochastic throughput evidence distinct."""
import hashlib
import json
from pathlib import Path
import statistics

REPO=Path(__file__).resolve().parents[3]
RAW=REPO/'benchmark-results/dflash2-serving-v031-20261009'


def main():
    names=['pro-w4-group5-matched96','pro-w4-group4-matched96']
    campaigns=[];controls=[];benchmarks=[];manifests=[]
    for name in names:
        root=RAW/name;c=json.loads((root/'campaign.json').read_text())
        assert c['status']=='PASS_W4_DFLASH_DIAGNOSTIC_SCREEN'
        assert c['active_context_limit']==96000 and c['depth']==7
        assert c['draft_sample_method']=='greedy' and c['draft_format']=='w4'
        assert c['restoration']['models_endpoint_healthy']
        assert c['restoration']['image_id']==c['production_image']
        controls.append(json.loads((root/'greedy-control.json').read_text())['controls'])
        audit=json.loads((root/'proposal-audit.json').read_text())
        assert audit['status']=='PASS_CAPTURED_PROPOSAL_DISTRIBUTION_AUDIT'
        assert all(x['gumbel_replay'] is not None for x in audit['rows'])
        benchmarks.append(json.loads((REPO/c['benchmark_results']).read_text()))
        campaigns.append(c)
        for path in [root/'campaign.json',root/'greedy-control.json',root/'proposal-audit.json',REPO/c['benchmark_results']]:
            manifests.append(dict(path=str(path.relative_to(REPO)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert campaigns[0]['draft']['output_sha256']==campaigns[1]['draft']['output_sha256']
    greedy=[]
    for a,b in zip(*controls):
        assert a['fixture']==b['fixture']
        for x,y in zip(a['responses'],b['responses']):
            assert x['mode']==y['mode'] and x['usage']==y['usage']
            for row in [x,y]:
                cached=row['metrics_delta'].get('vllm:prompt_tokens_cached_total',0)
                assert (cached>0)==(row['mode']=='warm'),('Unproven prefix mode',row['mode'],cached)
            ids_a,ids_b=x['token_ids'],y['token_ids']
            first=next((i for i,(u,v) in enumerate(zip(ids_a,ids_b)) if u!=v),None)
            counters=[]
            for row in [x,y]:
                d=row['metrics_delta'];rounds=d['vllm:spec_decode_num_drafts_total']
                assert rounds>0
                counters.append(dict(rounds=rounds,accepted_tokens=d['vllm:spec_decode_num_accepted_tokens_total'],
                    useful_counter_estimate=1+d['vllm:spec_decode_num_accepted_tokens_total']/rounds,
                    native_decode_time_per_counter_round_ms=d['vllm:request_decode_time_seconds_sum']/rounds*1000))
            greedy.append(dict(fixture=a['fixture'],mode=x['mode'],exact_tokens_equal=ids_a==ids_b,
                first_token_difference=first,group5_tps=x['native_decode_tps'],group4_tps=y['native_decode_tps'],
                relative_decode_gain=y['native_decode_tps']/x['native_decode_tps']-1,
                grouping_counter_summaries=counters,
                counter_note='Decode-time per round is a counter equivalent, not a GPU kernel measurement; token-limit tails and prefix-dependent proposals can affect it.'))
    stochastic=[]
    for name in ['phase-32k-c1','phase-64k-c1']:
        rates=[[x['native_weighted_decode_tokens_per_s'] for x in b['cases'] if x['scenario']['name']==name] for b in benchmarks]
        assert all(len(v)==3 for v in rates)
        medians=[statistics.median(v) for v in rates]
        stochastic.append(dict(name=name,all_rates=rates,medians=medians,median_ratio_gain=medians[1]/medians[0]-1))
    result=dict(status='MATCHED_96K_GROUPING_DIAGNOSTIC_NOT_PRODUCTION_PROMOTION',
        max_model_len=96000,greedy_controls=greedy,stochastic_cases=stochastic,
        deterministic_outputs_all_equal=all(x['exact_tokens_equal'] for x in greedy),
        raw_manifest=manifests,limits=['One arm per grouping, not a randomized replicated confidence study.',
            'Two short CPU-only audits overlapped group-five64K repeat2; cpu-reference-activity.json flags possible host interference. That sample is preserved.',
            'Greedy controls hold outputs fixed only where their token IDs agree.',
            'T1 repeats have different generated trajectories; report all repeats and acceptance.',
            '96K matched diagnostics do not themselves qualify192K speed or general semantic quality.'])
    out=REPO/'benchmarks/results/dflash2-pro-optimization-v031/grouping-matched96.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('MATCHED_GROUPING_ANALYZED',result['deterministic_outputs_all_equal'])


if __name__=='__main__':main()
