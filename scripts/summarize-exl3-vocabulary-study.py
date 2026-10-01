#!/usr/bin/env python3
"""Audit matched fixed-depth vocabulary serving and unique head ownership."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.campaign.resolve();assert not args.out.exists()
    state=json.loads((root/'campaign.json').read_text())
    assert state['status']=='COMPLETE' and state['depth']==3 and state['planned_waves']==24
    assert [a['vocabulary'] for a in state['arms']]==['pruned','full']
    assert state['arms'][0]['settings']==state['arms'][1]['settings']
    assert state['arms'][0]['settings']['speculative_config']=={'method':'mtp','num_speculative_tokens':3}
    tables=[];receipts=[];heads={}
    for arm in state['arms']:
        label=arm['vocabulary'];assert arm['status']=='COMPLETE' and len(arm['waves'])==12
        report=root/f'{label}-head-ownership.json'
        assert sha(report)==arm['head_ownership_sha256']
        h=json.loads(report.read_text());assert h==arm['head_ownership'] and h['shared_head_module']
        assert h['unique_pruned_storage_bytes']==(252315648 if label=='pruned' else 0)
        heads[label]=h;values={}
        for wave in arm['waves']:
            key=(wave['window'],wave['concurrency'],wave['cache_mode']);assert key not in values
            path=root/label/f'{key[0]}-c{key[1]}-{key[2]}'/'result.json'
            raw=json.loads(path.read_text());assert {k:v for k,v in raw.items() if k!='responses'}==wave
            assert raw['native']['completed']==key[1] and raw['native']['generation_tokens']==key[1]*512
            assert len(raw['responses'])==key[1]
            for response in raw['responses']:
                assert response['prompt_ids_verified'] and len(response['token_ids'])==512
                assert response['usage']['prompt_tokens']==wave['context_tokens']
                assert hashlib.sha256(json.dumps(response['token_ids'],separators=(',',':')).encode()).hexdigest()==response['output_ids_sha256']
            values[key]=raw;receipts.append({'path':str(path),'sha256':sha(path)})
        tables.append(values)
    wanted={(f'{domain}-4096',c,cache) for domain in ['code','prose'] for c in [1,4] for cache in ['cold','warm']}
    wanted|={('code-49152',4,cache) for cache in ['cold','warm']}
    wanted|={('code-102752',1,cache) for cache in ['cold','warm']}
    assert tables[0].keys()==tables[1].keys()==wanted
    rows=[]
    for key,a in tables[0].items():
        b=tables[1][key];metrics={}
        for name in ['native_request_weighted_decode_tps','client_aggregate_decode_tps','native_prefill_tps',
                     'wall_s','generated_tokens_per_round','joules_per_output_token']:
            metrics[name]={'pruned':a[name],'full':b[name],
                'full_change_pct':(b[name]/a[name]-1)*100 if a[name] and b[name] is not None else None}
        differences=[next((i for i,(x,y) in enumerate(zip(ra['token_ids'],rb['token_ids'])) if x!=y),None)
                     for ra,rb in zip(a['responses'],b['responses'])]
        rows.append({'window':key[0],'concurrency':key[1],'cache_mode':key[2],'metrics':metrics,
            'cache_hit_fraction':{label:w['native']['cached_tokens']/w['native']['prompt_tokens'] for label,w in [('pruned',a),('full',b)]},
            'native_accounting':{label:w['native'] for label,w in [('pruned',a),('full',b)]},
            'accepted_per_round_by_position':{label:w['accepted_per_round_by_position'] for label,w in [('pruned',a),('full',b)]},
            'first_output_difference_by_request':differences,'exact_output_matches':sum(v is None for v in differences)})
    requests=2*sum(r['concurrency'] for r in rows);assert requests==60
    assessment={'status':'COMPLETE_SCREEN_NOT_RELEASE_QUALIFICATION','campaign_sha256':sha(root/'campaign.json'),
        'image_id':state['image_id'],'depth':3,'requests':requests,'waves':24,'head_ownership':heads,
        'model_memory_saved_bytes':heads['pruned']['model_memory_usage']-heads['full']['model_memory_usage'],
        'paired_cases':rows,'raw_results':receipts,'comparison_design':state['comparison_design'],
        'scope':'Measured serving effects, including changed generated sequences and acceptance work. Warm reuse is observed; no isolated kernel-speed or numeric-quality claim. Head roles share one physical module; full vocabulary removes one unique pruned head.'}
    assert assessment['model_memory_saved_bytes']==252315648
    args.out.mkdir(parents=True);(args.out/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    lines=['# Fixed-MTP3 draft-vocabulary serving screen','',assessment['scope'],'',
        '| Context | C | Cache | Pruned decode tok/s | Full decode tok/s | Change | Cache hits pruned / full |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        m=row['metrics']['native_request_weighted_decode_tps' if row['concurrency']==1 else 'client_aggregate_decode_tps'];h=row['cache_hit_fraction']
        lines.append(f'| {row["window"]} | {row["concurrency"]} | {row["cache_mode"]} | {m["pruned"]:.2f} | {m["full"]:.2f} | {m["full_change_pct"]:+.1f}% | {h["pruned"]:.1%} / {h["full"]:.1%} |')
    lines+=['','C1 reports native request-weighted decode; C4 reports aggregate client decode intervals. Cold C4 can mix decode and prefill. Each arm is fresh; single pruned-then-full order has no ABBA/order confidence.',
        '',f'All {requests} requests complete. Full vocabulary saves252315648 bytes (~240.63MiB) of model allocation. This is one shared pruned-head allocation, not two independent copies. Actual KV capacity and output quality remain separate questions.','']
    (args.out/'README.md').write_text('\n'.join(lines));print(json.dumps({'status':assessment['status'],'requests':requests,'saved_bytes':assessment['model_memory_saved_bytes']}))


if __name__=='__main__':main()
