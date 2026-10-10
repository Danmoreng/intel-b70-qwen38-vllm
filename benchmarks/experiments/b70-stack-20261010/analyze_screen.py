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
    for root in a.additional:
        extra=json.loads((root/'campaign.json').read_text());assert extra['status']==d['status'] and extra['baseline']==d['baseline'] and extra['candidate']==d['candidate']
        d['blocks'].extend(extra['blocks'])
    cells={};weighted={m:[] for m in ['cold','warm']}
    for b in d['blocks']:
        maps={role:{(r['window'],r['mode']):r['result'] for r in b['arms'][role]} for role in ['baseline','candidate']}
        scores={m:0 for m in weighted}
        for key,x in maps['baseline'].items():
            y=maps['candidate'][key];name,mode=key
            cell=cells.setdefault(name+'/'+mode,dict(native_decode=[],client_aggregate=[],prefill=[],ttft=[],wall=[],acceptance=[]))
            for field,metric in [('native_decode','native_request_weighted_decode_tps'),('client_aggregate','client_aggregate_decode_tps'),('prefill','native_prefill_tps'),('wall','wall_s')]:
                cell[field].append(100*(y[metric]/x[metric]-1))
            cell['ttft'].append(100*(statistics.mean(y['client_ttft_s'])/statistics.mean(x['client_ttft_s'])-1))
            cell['acceptance'].append(dict(baseline=x['mean_acceptance_length'],candidate=y['mean_acceptance_length']))
            weight={'source-32k':.35,'source-64k':.35,'prose-32768':.30}.get(name,0)
            metric='client_aggregate_decode_tps' if name=='prose-32768' else 'native_request_weighted_decode_tps'
            scores[mode]+=weight*math.log(y[metric]/x[metric])
        for mode,v in scores.items():weighted[mode].append(100*math.expm1(v))
    result=dict(status='REVIEW_REQUIRED',images={k:d[k] for k in ['baseline','candidate']},independent_blocks=len(d['blocks']),
        cells={key:{k:summarize(v) if k!='acceptance' else v for k,v in cell.items()} for key,cell in cells.items()},
        weighted_decode={mode:summarize(x) for mode,x in weighted.items()},
        definitions='Within each paired block: weighted geometric ratio, .35 source32 C1 native + .35 source64 C1 native + .30 prose32 C4 client aggregate. Cold and warm separate. Positive wall/TTFT percentages are worse; positive TPS percentages are better.',
        limits='Small independent block sample. Percentile paired bootstrap is descriptive and does not make three starts a high-powered noninferiority study. Fresh first-start compiler effects retained. Protected cells and full final matrix remain separate gates; no promotion from this file alone.')
    result['performance_survivor']=all(x['mean_pct']>=3 and x['bootstrap_95_pct'][0]>0 for x in result['weighted_decode'].values())
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['weighted_decode']),flush=True)
if __name__=='__main__':main()
