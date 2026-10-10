"""Paired uncertainty uses independent startup blocks, never tokens as samples."""
import argparse,json,math,random,statistics
from pathlib import Path

def ci(x):
    rng=random.Random(20261010);n=len(x)
    d=sorted(statistics.mean(rng.choices(x,k=n)) for _ in range(20000))
    return [d[499],d[19499]]
def summarize(x):
    return dict(block_pct=x,mean_pct=statistics.mean(x),bootstrap_95_pct=ci(x))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--additional',type=Path,action='append',default=[]);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    d=json.loads((a.root/'campaign.json').read_text());assert d['status']=='COMPLETE_PAIRED_SCREEN_REVIEW_REQUIRED'
    for b in d['blocks']:b['_root']=a.root
    for root in a.additional:
        extra=json.loads((root/'campaign.json').read_text());assert extra['status']==d['status'] and extra['baseline']==d['baseline'] and extra['candidate']==d['candidate']
        for b in extra['blocks']:b['_root']=root
        d['blocks'].extend(extra['blocks'])
    cells={};weighted={m:[] for m in ['cold','warm']}
    for b in d['blocks']:
        maps={role:{(r['window'],r['mode']):r['result'] for r in b['arms'][role]} for role in ['baseline','candidate']}
        required={(name,mode) for name in ['source-32k','source-64k','prose-32768'] for mode in ['cold','warm']}
        assert all(required<=set(m) for m in maps.values()),'Incomplete weighted panel; a first-touch diagnostic is not a promotion screen'
        scores={m:0 for m in weighted}
        for key,x in maps['baseline'].items():
            y=maps['candidate'][key];name,mode=key
            cell=cells.setdefault(name+'/'+mode,dict(native_decode=[],client_aggregate=[],prefill=[],ttft=[],wall=[],e2e_slowest_request=[],acceptance=[],client_delivery=[]))
            for field,metric in [('native_decode','native_request_weighted_decode_tps'),('client_aggregate','client_aggregate_decode_tps'),('prefill','native_prefill_tps'),('wall','wall_s')]:
                cell[field].append(100*(y[metric]/x[metric]-1))
            cell['ttft'].append(100*(statistics.mean(y['client_ttft_s'])/statistics.mean(x['client_ttft_s'])-1))
            cell['acceptance'].append(dict(baseline=x['mean_acceptance_length'],candidate=y['mean_acceptance_length']))
            delivery={}
            for role,value in [('baseline',x),('candidate',y)]:
                path=b['_root']/f"block-{b['index']}-{role}"/name/f"{name}-c{value['concurrency']}-{mode}"/'result.json'
                requests=json.loads(path.read_text())['responses']
                delivery[role]=[dict(e2e_s=r['wall_s'],ttft_s=r['ttft_client_s'],
                    tpot_s=(r['wall_s']-r['ttft_client_s'])/(r['usage']['completion_tokens']-1),
                    delivery_gaps_s=r['client_token_delivery_gaps_s'],
                    output_ids_sha256=r['output_ids_sha256'],output_text_sha256=r['output_text_sha256']) for r in requests]
            cell['e2e_slowest_request'].append(100*(max(r['e2e_s'] for r in delivery['candidate'])/max(r['e2e_s'] for r in delivery['baseline'])-1))
            cell['client_delivery'].append(delivery)
            weight={'source-32k':.35,'source-64k':.35,'prose-32768':.30}.get(name,0)
            metric='client_aggregate_decode_tps' if name=='prose-32768' else 'native_request_weighted_decode_tps'
            scores[mode]+=weight*math.log(y[metric]/x[metric])
        for mode,v in scores.items():weighted[mode].append(100*math.expm1(v))
    result=dict(status='REVIEW_REQUIRED',images={k:d[k] for k in ['baseline','candidate']},independent_blocks=len(d['blocks']),
        cells={key:{k:summarize(v) if k not in ('acceptance','client_delivery') else v for k,v in cell.items()} for key,cell in cells.items()},
        weighted_decode={mode:summarize(x) for mode,x in weighted.items()},
        definitions='Within each paired block: weighted geometric ratio, .35 source32 C1 native + .35 source64 C1 native + .30 prose32 C4 client aggregate. Cold and warm separate. Positive wall/TTFT percentages are worse; positive TPS percentages are better.',
        limits='Small independent block sample. Percentile paired bootstrap is descriptive and does not make three starts a high-powered noninferiority study. Fresh first-start compiler effects retained. Protected cells and full final matrix remain separate gates; no promotion from this file alone.')
    result['performance_survivor']=all(x['mean_pct']>=3 and x['bootstrap_95_pct'][0]>0 for x in result['weighted_decode'].values())
    result['tail_scope']='Paired uncertainty for the slowest client request in each wave, not a high-powered population p99 claim. Per-request TPOT and SSE delivery-gap percentiles are retained; SSE gaps are not individual GPU-step ITL.'
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['weighted_decode']),flush=True)
if __name__=='__main__':main()
