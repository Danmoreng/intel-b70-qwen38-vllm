#!/usr/bin/env python3
"""Validate complete passed shape classes from terminal, attested row probes."""
import argparse
import hashlib
import json
from pathlib import Path

ROWS=[1,2,3,4,5,8,12,16,20,24,32,40,48,64,128,129,256,512]
COMPILED=[1,4,20,64,128,129,512]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--heads',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists()
    a=json.loads(args.base.read_text());b=json.loads(args.heads.read_text())
    assert a['status']=='FAILED' and a['error'].startswith('FailOnRecompileLimitHit(')
    assert b['status']=='PASS_EXPANDED_ROW_PADDING_GRAPH_GATE'
    assert a['native_library_sha256']==b['native_library_sha256']=='48f879c16f3695dbb0fae7388f1e318591d6408acc7be15b0093a0bbb6d9053b'
    assert a['checkpoint_index_sha256']==b['checkpoint_index_sha256']
    groups=[]
    for source,names in [(a,['gdn-qkvz','attention-qkv','gate-up','down']),
                         (b,['head-first','head-last','head-full'])]:
        for name in names:
            matches=[v for v in source['groups'] if v['name']==name];assert len(matches)==1
            row=matches[0]
            assert [v['rows'] for v in row['cases']]==ROWS
            assert [v['actual'] for v in row['padding']]==ROWS
            assert [v['rows'] for v in row['compiled']]==COMPILED
            for v in row['cases']:
                assert v['reference']['finite'] and v['reference']['relative_norm']<.002
                if v['rows']<=128:
                    for key in ('graph','mutated_graph'):
                        assert v[key]['finite'] and v[key]['relative_norm']<.002
                else:assert v['graph_applicability'].startswith('Unsupported oneDNN INT8-prefill capture;')
            for v in row['padding']:
                for key in ('tail_poison','same_family_active'):
                    assert v[key]['finite'] and v[key]['relative_norm']<.002
            for v in row['compiled']:
                assert v['after_large_first']['finite'] and v['after_large_first']['relative_norm']<.002
            groups.append(row)
    result={k:b[k] for k in ('torch','seed','rows','tolerance','native_library_sha256','checkpoint_index_sha256')}
    result.update(status='PASS_EXPANDED_ROW_PADDING_GRAPH_GATE',groups=groups,
        source_evidence={str(args.base):sha(args.base),str(args.heads):sha(args.heads)},
        qualification_contract='Frozen FULL_DECODE_ONLY graph profile, maximum64 target rows. Native SmallM graphs tested additionally through128. M129/256/512 are eager INT8-prefill; unsupported large-prefill capture remains explicitly outside the production contract.',
        reuse_note='Four complete passed classes reused from a terminal probe whose later head closure hit the harness Dynamo recompile limit. No failed/incomplete head data is reused. Head classes rerun with isolated compiler state, including all248320 columns.')
    args.out.parent.mkdir(parents=True);args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
