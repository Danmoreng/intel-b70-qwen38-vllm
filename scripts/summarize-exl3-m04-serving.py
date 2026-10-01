#!/usr/bin/env python3
"""Audit complete M04 candidate serving against exact fixed-MTP3 raw baseline."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.campaign.resolve();assert not args.out.exists()
    state=json.loads((root/'campaign.json').read_text())
    assert state['status']=='COMPLETE_SERVING_SCREEN_NOT_RELEASE_QUALIFICATION'
    assert len(state['waves'])==len(state['baseline_waves'])==8
    assert {k:v for k,v in state['settings'].items() if k!='worker_extension_cls'}=={k:v for k,v in state['baseline_settings'].items() if k!='worker_extension_cls'}
    assert state['settings']['speculative_config']=={'method':'mtp','num_speculative_tokens':3}
    logs=(root/'worker.log').read_text()
    assert 'EXL3_M04_INSTALL_PASS' in logs
    assert all(f'EXL3_M04_DISPATCH ({b}, 4, 1600, True)' in logs for b in [1,4]),'M04 not proven in C1/C4 graph capture'
    rows=[];receipts=[]
    for baseline,wave in zip(state['baseline_waves'],state['waves']):
        key=(wave['window'],wave['concurrency'],wave['cache_mode'])
        assert key==(baseline['window'],baseline['concurrency'],baseline['cache_mode'])
        paths=[Path(baseline['reused_result']),root/f'{key[0]}-c{key[1]}-{key[2]}'/'result.json']
        assert sha(paths[0])==baseline['reused_result_sha256']
        values=[]
        for path,summary in zip(paths,[{k:v for k,v in baseline.items() if not k.startswith('reused_result')},wave]):
            value=json.loads(path.read_text());assert {k:v for k,v in value.items() if k!='responses'}==summary
            assert value['native']['completed']==key[1] and len(value['responses'])==key[1]
            assert value['native']['generation_tokens']==key[1]*512
            for response in value['responses']:
                assert response['prompt_ids_verified'] and len(response['token_ids'])==512
                assert response['usage']['prompt_tokens']==wave['context_tokens']
                assert hashlib.sha256(json.dumps(response['token_ids'],separators=(',',':')).encode()).hexdigest()==response['output_ids_sha256']
            values.append(value);receipts.append({'path':str(path),'sha256':sha(path)})
        a,b=values;metrics={}
        for name in ['native_request_weighted_decode_tps','client_aggregate_decode_tps','native_prefill_tps',
                     'wall_s','generated_tokens_per_round','joules_per_output_token']:
            metrics[name]={'native':a[name],'m04':b[name],'m04_change_pct':(b[name]/a[name]-1)*100 if a[name] and b[name] is not None else None}
        differences=[next((i for i,(x,y) in enumerate(zip(ra['token_ids'],rb['token_ids'])) if x!=y),None)
                     for ra,rb in zip(a['responses'],b['responses'])]
        rows.append({'window':key[0],'concurrency':key[1],'cache_mode':key[2],'metrics':metrics,
            'cache_hit_fraction':{label:v['native']['cached_tokens']/v['native']['prompt_tokens'] for label,v in [('native',a),('m04',b)]},
            'native_accounting':{label:v['native'] for label,v in [('native',a),('m04',b)]},
            'first_output_difference_by_request':differences,'exact_matches':sum(v is None for v in differences)})
    assert sum(r['concurrency'] for r in rows)==20
    result={'status':'COMPLETE_SERVING_SCREEN_NOT_RELEASE_QUALIFICATION','image_id':state['image_id'],
        'campaign_sha256':sha(root/'campaign.json'),'worker_log_sha256':sha(root/'worker.log'),
        'library_sha256':state['library_sha256'],'source_sha256':state['source_sha256'],'pairs':rows,'raw_results':receipts,
        'design':state['design'],'scope':'Actual graph-path M04 dispatch proven at capture; serving includes acceptance/output variation and mixed prefill. Numerical micro gates pass separately. Final generated-route quality, mixed tails, stability and release gates remain.'}
    args.out.mkdir(parents=True);(args.out/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Shared-KV verification serving screen','',result['scope'],'',
        '| Context | C | Cache | Native decode tok/s | M04 decode tok/s | Change | Cache hits native / M04 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        m=row['metrics']['native_request_weighted_decode_tps' if row['concurrency']==1 else 'client_aggregate_decode_tps'];h=row['cache_hit_fraction']
        lines.append(f'| {row["window"]} | {row["concurrency"]} | {row["cache_mode"]} | {m["native"]:.2f} | {m["m04"]:.2f} | {m["m04_change_pct"]:+.1f}% | {h["native"]:.1%} / {h["m04"]:.1%} |')
    lines+=['',state['design'],'','C1 uses native request-weighted decode; C4 uses aggregate client decode intervals. These are serving observations, not isolated kernel rates. Exact output divergence and actual cache reuse are retained. No production change follows from this screen.','']
    (args.out/'README.md').write_text('\n'.join(lines));print(json.dumps({'status':result['status'],'paired_requests':20}))


if __name__=='__main__':main()
