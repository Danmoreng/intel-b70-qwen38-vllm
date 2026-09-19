"""Paired short serving screens. Separate cold/warm native phase accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time

from common import Session,recover,diag,save,reset,ROOT,REPO,BASE,load

def summarize(rows):
    summary=[]
    for context in sorted({r['context'] for r in rows}):
        for cache in ('cold','warm'):
            selected=[r for r in rows if r['context']==context and r['condition']==cache]
            groups={}
            for r in selected:groups.setdefault(r['block'],{})[r['arm']]=r
            pairs=[p for p in groups.values() if set(p)=={'control','candidate'}]
            for metric in ('native_prefill_seconds','native_decode_seconds','ttft_s','total_s'):
                advantages=[100*(1-p['candidate'][metric]/p['control'][metric]) for p in pairs]
                if not advantages:continue
                rng=random.Random(190919);boot=sorted(statistics.mean(rng.choices(advantages,k=len(advantages))) for _ in range(5000))
                summary.append({'context':context,'cache':cache,'metric':metric,'pairs':len(pairs),
                    'paired_latency_reduction_pct':statistics.mean(advantages),
                    'bootstrap_95pct':[boot[125],boot[4874]],'pair_values':advantages})
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['e01','recover'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--blocks',type=int,default=5)
    args=parser.parse_args();run=args.run_dir.resolve()
    if args.mode=='recover':recover(run);return
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    run.mkdir(parents=True,exist_ok=True)
    if args.mode=='e01':
        real=Path((ROOT/'e01/LATEST-REAL').read_text().strip())
        assert json.loads((real/'real-decision.json').read_text())['pass']
    shutil.copyfile(ROOT/'fixtures/fixtures.json',run/'fixtures.json')
    shutil.copyfile(REPO/'scripts/context-benchmark.py',run/'context-benchmark.py')
    harness=load('frozen_context',run/'context-benchmark.py')
    fixtures=json.loads((run/'fixtures.json').read_text())
    rows=[];save(run/'manifest.json',{'experiment':args.mode,'blocks':args.blocks,'contexts':[8192,16384],
        'output_tokens':256,'sampling':{'temperature':0},'production_sampling_track':'quality requests separately',
        'baseline':BASE,'candidate_image':'local/qwen38-b70-vllm:e01-qk-rope-gate-20260919' if args.mode=='e01' else BASE,
        'fixture_sha256':hashlib.sha256((run/'fixtures.json').read_bytes()).hexdigest(),
        'harness_sha256':hashlib.sha256((run/'context-benchmark.py').read_bytes()).hexdigest(),
        'gate':'At least 2% paired affected-phase benefit, CI positive, no systematic >2% other-phase loss; all correctness/context gates required before promotion.'})
    with Session(run) as session:
        for block in range(args.blocks):
            order=('control','candidate') if block%2==0 else ('candidate','control')
            for arm in order:
                flag=int(arm=='candidate' and args.mode=='e01')
                budget=6144 if args.mode=='e02' and arm=='candidate' else 4096
                image='local/qwen38-b70-vllm:e01-qk-rope-gate-20260919' if args.mode=='e01' else BASE
                label=f'block-{block}-{arm}'
                save(run/'progress.json',{'phase':'starting','block':block,'arm':arm,'blocks':args.blocks})
                profiler={'profiler':'torch','torch_profiler_dir':'/evidence/trace',
                    'torch_profiler_with_stack':False,'torch_profiler_record_shapes':True,
                    'torch_profiler_dump_cuda_time_total':False,'ignore_frontend':True}
                out=session.start(label,image,flags={'B70_FUSED_QK_ROPE_GATE':str(flag)},budget=budget,
                    extra=['--profiler-config',json.dumps(profiler)])
                def request(item,condition,tag,output=256,prefix=False,sampling=None):
                    target=item['prefix_target'] if prefix else item['target']
                    messages=item['prefix_messages'] if prefix else item['messages']
                    record=harness.request(diag.EXPERIMENT_URL+'/v1/chat/completions',diag.EXPERIMENT_URL,
                        'Qwen3.8-27B',messages,output,str(out/(tag+'.sse.jsonl')),condition,
                        expected_prompt_tokens=target,ignore_eos=True,sampling=sampling)
                    assert record['completion_tokens']==output
                    assert record['preemptions']==0,'Recompute invalidates this paired screen'
                    save(out/(tag+'.json'),record)
                    return record
                # Include output graph shapes in warmup; never count warmup tokens.
                for context in (8192,16384):
                    warm=next(f for f in fixtures if f['target']==context and f['rep']==0)
                    reset();request(warm,'cold',f'warmup-{context}')
                for context in (8192,16384):
                    fixture=next(f for f in fixtures if f['target']==context and f['rep']==block+1)
                    for condition in ('cold','warm'):
                        save(run/'progress.json',{'phase':'measuring','block':block,'arm':arm,'context':context,'cache':condition})
                        reset()
                        # A one-token request can finish before the hybrid/MTP
                        # path retains reusable prefix state. Exercise decode
                        # during priming and require observed hits afterwards.
                        if condition=='warm':request(fixture,'cold',f'prime-{context}',output=32,prefix=True)
                        r=request(fixture,condition,f'{condition}-{context}')
                        assert r['prefix_cache_hits_delta']>0 if condition=='warm' else r['computed_prompt_tokens']==context
                        r.update({'block':block,'arm':arm,'context':context,'condition':condition,'budget':budget})
                        rows.append(r)
                        with (run/'serving-results.jsonl').open('a') as stream:stream.write(json.dumps(r)+'\n')
                        save(run/'paired-summary.json',summarize(rows))
                if block==0:
                    # Small semantics suites and teacher-forced samples, outside timing.
                    save(run/'progress.json',{'phase':'quality','block':block,'arm':arm})
                    with (out/'quality.log').open('w') as log:
                        subprocess.run(['python3',str(ROOT/'quality.py'),
                            '--root',diag.EXPERIMENT_URL,'--image',BASE,'--output-dir',str(out/'quality')],
                            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                    assert json.loads((out/'quality/results.json').read_text())['coding_pass']
                    subprocess.run(['python3','/home/sebastian/LocalLLM/Local-AI-B70/qwen38/production/check-vision-tools.py',
                        '--root',diag.EXPERIMENT_URL,'--output',str(out/'vision-tools.json')],check=True,timeout=900)
                    subprocess.run(['python3',str(REPO/'scripts/check-prefix-state.py'),'--root',diag.EXPERIMENT_URL,
                        '--output',str(out/'prefix-state.json')],check=True,timeout=900)
                    item=next(f for f in fixtures if f['target']==8192 and f['rep']==1)
                    reset();request(item,'cold','production-sampling',sampling={'temperature':1,'top_p':.95,'top_k':20,'seed':190919})
                    # One short diagnostic trace, never mixed into timing records.
                    reset();diag.api('/start_profile',{})
                    request(item,'cold','profile-excluded',output=32)
                    diag.api('/stop_profile',{},timeout=120)
                session.stop()
        summary=summarize(rows);save(run/'paired-summary.json',summary)
        prefill=[s for s in summary if s['metric']=='native_prefill_seconds']
        decode=[s for s in summary if s['metric']=='native_decode_seconds']
        def winner(phase):
            return len(phase)==4 and all(s['pairs']>=5 and s['paired_latency_reduction_pct']>=2
                                        and s['bootstrap_95pct'][0]>0 for s in phase)
        go=(winner(prefill) or (args.mode=='e01' and winner(decode))) and all(
            s['paired_latency_reduction_pct']>=-2 for s in prefill+decode)
        save(run/'short-gate.json',{'pass':go,'status':'needs-extended-qualification' if go else 'no-promotion',
            'scope':'paired 8K/16K only; no 32K/64K/full-context/coding qualification yet'})
        save(run/'progress.json',{'phase':'short-screen-complete','go':go})

if __name__=='__main__':main()
