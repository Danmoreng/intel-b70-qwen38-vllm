"""B8/V4 conditional screening calculation from actual prefix counters."""
import hashlib
import json
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[3]


def counters(path):
    values={}
    for line in path.read_text().splitlines():
        match=re.fullmatch(r'([^\s{]+)(\{[^}]*\})?\s+([0-9.eE+-]+)',line)
        if not match:continue
        name,labels,value=match.groups()
        if name=='vllm:spec_decode_num_accepted_tokens_per_pos_total':
            pos=re.search(r'position="(\d+)"',labels or '');assert pos
            name+=':'+pos[1]
        values[name]=values.get(name,0)+float(value)
    return values


def calculate(name):
    folder=REPO/'benchmark-results/dflash2-serving-v031-20261009'/name
    campaign=json.loads((folder/'campaign.json').read_text())
    root=(REPO/campaign['benchmark_results']).parent
    results=[]
    for scenario in ['phase-32k-c1','phase-64k-c1']:
        total=0;accepted=[0]*7;sources=[]
        for repeat in range(1,4):
            p=root/f'{scenario}-r{repeat}'
            before=counters(p/'metrics-before.prom');after=counters(p/'metrics-after.prom')
            rounds=after['vllm:spec_decode_num_drafts_total']-before.get('vllm:spec_decode_num_drafts_total',0)
            positions=[after[f'vllm:spec_decode_num_accepted_tokens_per_pos_total:{i}']-before.get(f'vllm:spec_decode_num_accepted_tokens_per_pos_total:{i}',0) for i in range(7)]
            assert rounds>0 and rounds>=positions[0] and all(a>=b for a,b in zip(positions,positions[1:]))
            total+=rounds;accepted=[a+b for a,b in zip(accepted,positions)]
            for f in ['metrics-before.prom','metrics-after.prom']:
                source=p/f;sources.append(dict(path=str(source.relative_to(REPO)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        u7=1+sum(accepted)/total;u4=1+sum(accepted[:4])/total
        results.append(dict(scenario=scenario,rounds=total,accepted_prefix_counts=accepted,
            useful_tokens_per_round_v7=u7,truncated_prefix_estimate_v4=u4,
            required_fraction_round_time_saved=1-u4/u7,raw_sources=sources))
    return dict(campaign=name,results=results)


if __name__=='__main__':
    result=dict(status='CONDITIONAL_COST_SCREEN_NOT_B8V4_MEASUREMENT',
        arms=[calculate(n) for n in ['pro-w4-k7-attempt3','pro-w4-group4-k7']],
        limits=['Truncating counters assumes the same proposal law; this is not a measured B8/V4 acceptance result.',
            'A new V changes subsequent target prefixes, scheduling and potentially proposal distributions.',
            'Target EXL3 DPAS uses MB8 for both five and eight rows. The unchanged eight-row draft backbone is not eliminated.',
            'M04 uses query tile8 for both. At32K both use16 splits; at64K the eight-row path adapts to32 and the five-row path uses16, so equal-cost assumptions there need measurement.',
            'The MB8 argument is specific to C1. The pinned native build enables MB24: at C4 target rows20 versus32 use MB24 versusMB32. A C4 saving is possible and must not be ruled out using the C1 tile argument.',
            'Kernel event medians and aggregate acceptance counters must not be combined as if they came from one exact round.'])
    output=REPO/'benchmarks/results/dflash2-pro-optimization-v031/verification-cost.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    for arm in result['arms']:
        print(arm['campaign'],[(x['scenario'],round(100*x['required_fraction_round_time_saved'],2)) for x in arm['results']])
