#!/usr/bin/env python3
"""Bounded exact-image release gates, sequentially using the whole GPU.

The image/profile/native artifact stay frozen. Failed checks restore EXL3 v1.
No coding campaign, original-model inference, kernel or vocabulary tuning.
"""
import argparse
import concurrent.futures
import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from exl3_candidate_worker import BASE, MODEL, REPO, Worker, profile, sha
from runpy import run_path


def module(name,file):
    s=importlib.util.spec_from_file_location(name,REPO/'scripts'/file)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


S=module('review_serving','run-exl3-review-serving.py')
M=module('review_mtp','run-exl3-mtp-study.py')
X=module('review_mixed','run-exl3-mixed-qualification.py')
D=module('review_contract','run-exl3-contract-diagnostics.py')
O=module('review_operations','run-exl3-final-operations.py')


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-dir',type=Path,required=True)
    p.add_argument('--serving',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--leave-offline',action='store_true')
    p.add_argument('--phases',default='operations,matched,quality,telemetry')
    p.add_argument('--reuse-passed-from',type=Path,help='Retain completed same-image phases from a previous campaign; never reuse a failed phase')
    a=p.parse_args();root=a.out.resolve();assert not root.exists();root.mkdir(parents=True)
    phases=a.phases.split(',');assert len(set(phases))==len(phases) and set(phases)<=set(['operations','matched','quality','telemetry'])
    directory=a.release_dir.resolve();release=read(directory/'production_image.json');image=release['image_id']
    saved=read(REPO/'config/releases/exl3-v1/production_image.json');baseline=saved['image_id']
    assert image=='sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918'
    assert release['environment_overrides']=={'EXL3_SDPA_CACHE_CAPACITY':'64'}
    policy=directory/'production_policy.json';assert sha(policy)==release['policy_sha256']
    panel_path=REPO/'benchmark-results/exl3-optimized-quality-v2/performance-panel.json.gz'
    panel=json.loads(gzip.decompress(panel_path.read_bytes()))
    fixture=REPO/'benchmarks/experiments/exl3-review-release-v2'
    state=dict(status='WAITING_EXCLUSIVE_GPU',started_unix=time.time(),image_id=image,
        baseline_image_id=baseline,policy_sha256=release['policy_sha256'],
        environment_overrides=release['environment_overrides'],performance_panel_sha256=sha(panel_path),
        sources_sha256={str(f.relative_to(REPO)):sha(f) for f in [Path(__file__),REPO/'scripts/exl3_candidate_worker.py',*fixture.glob('*.py')]},phases={})
    if a.reuse_passed_from:
        previous=a.reuse_passed_from.resolve();prior=read(previous/'campaign.json')
        assert prior['status'] in ('FAILED','PASS_BOUNDED_REVIEW_RELEASE_QUALIFICATION')
        for key in ['image_id','baseline_image_id','policy_sha256','environment_overrides','performance_panel_sha256']:
            assert prior[key]==state[key], 'Reuse identity mismatch: '+key
        assert prior['serving_campaign_sha256']==sha(a.serving/'campaign.json')
        reused=[]
        for phase,value in prior['phases'].items():
            if phase in phases:continue
            assert value['status']=='PASS', 'Cannot reuse a failed phase: '+phase
            (root/phase).symlink_to(previous/phase,target_is_directory=True)
            state['phases'][phase]=dict(value,reused=True);reused.append(phase)
        state['reused_passed_phases']=dict(campaign_path=str((previous/'campaign.json').relative_to(REPO)),campaign_sha256=sha(previous/'campaign.json'),phases=reused,
            prior_status=prior['status'],prior_error=prior.get('error'),scope='Only completed same-image operating, matched and quality phases retained. Failed diagnostic setup is not a passed check.')
    def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    def worker(path,role='candidate',**kwargs):
        return Worker(image if role=='candidate' else baseline,path,
            name='b70-exl3-review-qualification',env={'EXL3_SDPA_CACHE_CAPACITY':'64',**kwargs.pop('env',{})} if role=='candidate' else kwargs.pop('env',{}),**kwargs)
    def command(name,argv):
        state['phases']['quality']['commands'].append(dict(name=name,argv=argv));save()
        with (root/'quality'/(name+'.log')).open('w') as log:
            subprocess.run(argv,check=True,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    def native_command(script,args):
        cfg=profile(image); common=['docker','run','--rm','--name','b70-exl3-review-quality','--network','none','--device','/dev/dri','--memory','14g','--shm-size','4g',
            '-v',str(MODEL)+':/models/checkpoint:ro','-v',str(REPO/'benchmarks/experiments/quantization-reference')+':/scripts:ro',
            '-v',str(fixture)+':/review:ro','-v',str(root/'quality')+':/results',
            '-v',str(REPO/'benchmark-results/quantization-reference-20261001')+':/reference:ro',
            '-v',str(REPO/'benchmark-results/exl3-long-quality-v3/bf16')+':/long-reference:ro',
            '-v',str(REPO/'benchmark-results/exl3-optimized-quality-v2/short/target-optimized')+':/v1-short:ro']
        env={**cfg['env'],'EXL3_SDPA_CACHE_CAPACITY':'64','HF_HUB_OFFLINE':'1','PYTHONPATH':'/scripts:/review','ZE_AFFINITY_MASK':'0','OMP_NUM_THREADS':'4','EXL3_LOADER_REPORT_DIR':'/results/loader-'+script.replace('.py','')}
        cache=Path.home()/'.cache/exl3xpu/review-release-quality'/image.removeprefix('sha256:')
        for k,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
            (cache/k).mkdir(parents=True,exist_ok=True);common+=['-v',str(cache/k)+':'+target]
        for k,v in env.items():common+=['-e',k+'='+v]
        return common+['--entrypoint','python',image,'-u',script,*args]
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        restore_image=S.current_qualified_image();state['restore_image_id']=restore_image
        assert read(a.serving/'campaign.json')['status']=='COMPLETE_REVIEW_SERVING_MATRIX'
        assert read(a.serving/'campaign.json')['image_receipt']['image_id']==image
        state['serving_campaign_sha256']=sha(a.serving/'campaign.json');state['status']='RUNNING';save()
        D.R.cap();subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
        success=False
        try:
            for phase in phases:
                path=root/phase;path.mkdir();state['phases'][phase]=dict(status='RUNNING');save();print('START',phase,flush=True)
                if phase=='operations':
                    w=worker(path/'serving',env={'EXL3_MIGRATION_TRACE':'/results/trace'})
                    try:
                        w.start();report=dict(loader=S.loader(w.root),cases={})
                        for name,fn in [('api',lambda p:D.smoke(p,262144,{'image':32,'video':4})),
                            ('media',lambda p:D.media_case(p,{'image':32,'video':4})),
                            ('near-limit',lambda p:D.long_case('near-limit',p,262144)),
                            *[(name,lambda p,n=name:D.operational(n,p,True)) for name in ['c4-long','c16-long','image-long','extension-abort-long']]]:
                            destination=path/name;destination.mkdir();print('CASE',name,flush=True)
                            report['cases'][name]=fn(destination)
                            if name=='near-limit':assert report['cases'][name]['native']['preemptions']==0
                            (path/'operations.json').write_text(json.dumps(report,indent=2)+'\n')
                        pictures=[O.unique_red(i) for i in range(32)]
                        before=D.R.snapshot(BASE)
                        response=D.R.http(BASE,'/v1/chat/completions',D.chat(pictures+[dict(type='text',text='What main color is shared by all32 pictures? Reply with one word.')]),timeout=1800)
                        _,native=D.R.wait_accounted(BASE,before,response['usage'])
                        assert 'red' in response['choices'][0]['message']['content'].lower() and native['preemptions']==0
                        assert response['usage']['prompt_tokens']>100000
                        report['cases']['independent32-max-area-images']=dict(status='PASS',usage=response['usage'],native=native,unique_images=32,pixels_each=4194304,
                            fixture_sha256=[hashlib.sha256(x['image_url']['url'].encode()).hexdigest() for x in pictures])
                        (path/'operations.json').write_text(json.dumps(report,indent=2)+'\n')
                        subprocess.run([sys.executable,str(REPO/'scripts/collect-review-runtime.py'),'--container',w.name,'--output',str(path/'runtime-sources')],check=True)
                    finally:w.stop()
                    aborted=sum(p.read_text().count('FINISHED_ABORTED') for p in (w.root/'trace').glob('*') if p.is_file() and p.suffix in ('.json','.jsonl'))
                    assert aborted>0;report['confirmed_abort_trace_occurrences']=aborted
                    w=worker(path/'restarted')
                    try:
                        w.start();response=D.R.http(BASE,'/v1/chat/completions',D.chat('Reply with the capital of France, one word.'),timeout=180)
                        assert 'paris' in response['choices'][0]['message']['content'].lower()
                        report['cases']['worker-restart']=dict(status='PASS',loader=S.loader(w.root),reply=response['choices'][0]['message']['content'])
                    finally:w.stop()
                    report['status']='PASS_EXACT_IMAGE_OPERATIONAL_GATES';(path/'operations.json').write_text(json.dumps(report,indent=2)+'\n')
                elif phase=='matched':
                    cases=[('short-c1','code-4096',1),('long-c1','code-102752',1),('short-c4','code-4096',4),('moderate-c4','code-32768',4)]
                    arms=[]
                    def arm(role,index,selected,mixed):
                        arm_path=path/f'arm-{index}-{role}';w=worker(arm_path,role);d=dict(role=role,image_id=w.image,waves=[],mixed=None);arms.append(d)
                        try:
                            w.start();short=next(x for x in panel['windows'] if x['name']=='code-4096')
                            for c in (1,4):M.wave(short,c,'warmup',0,3,arm_path,64)
                            for label,name,c in selected:
                                win=next(x for x in panel['windows'] if x['name']==name)
                                print('MATCHED',role,label,flush=True)
                                result=M.wave(win,c,'cold',0,3,arm_path,512)
                                assert result['native']['preemptions']==0;d['waves'].append(dict(label=label,result=result))
                            if mixed:d['mixed']=X.mixed(panel,3,arm_path/'mixed-c4',0)
                        finally:w.stop()
                        (path/'arms.json').write_text(json.dumps(arms,indent=2)+'\n')
                        return d
                    def compare(v1,candidate):
                        rows=[]
                        for left,right in zip(v1['waves'],candidate['waves']):
                            assert left['label']==right['label'];x,y=left['result'],right['result']
                            decode=100*(y['native_request_weighted_decode_tps']/x['native_request_weighted_decode_tps']-1)
                            ttft=100*(statistics.median(y['client_ttft_s'])/statistics.median(x['client_ttft_s'])-1)
                            rows.append(dict(label=left['label'],decode_change_pct=decode,ttft_change_pct=ttft,trigger=decode < -5 or ttft>5))
                        if v1['mixed'] and candidate['mixed']:
                            x,y=v1['mixed'],candidate['mixed']
                            ttft=100*(y['incoming_prefill_client_s']/x['incoming_prefill_client_s']-1)
                            p95=100*(y['during_incoming_prefill_burst_gaps_s']['p95']/x['during_incoming_prefill_burst_gaps_s']['p95']-1)
                            rows.append(dict(label='mixed-c4',ttft_change_pct=ttft,p95_stream_gap_change_pct=p95,trigger=ttft>5 or p95>10))
                        return rows
                    v1=arm('v1',0,cases,True);candidate=arm('candidate',1,cases,True);initial=compare(v1,candidate)
                    triggered={x['label'] for x in initial if x['trigger']};repeat=[]
                    if triggered:
                        selected=[c for c in cases if c[0] in triggered]
                        b=arm('candidate',2,selected,'mixed-c4' in triggered);a1=arm('v1',3,selected,'mixed-c4' in triggered);repeat=compare(a1,b)
                    report=dict(status='PASS_MATCHED_SERVING_SCREEN' if not any(x['trigger'] for x in repeat) else 'UNRESOLVED_REPRODUCIBLE_REGRESSION',
                        initial=initial,repeat=repeat,arms=arms,
                        criteria='Review triggers: decode slower >5%, TTFT slower >5%, mixed p95 SSE gap higher >10%; repeat only affected case once. Finite deterministic workload, not an SLA or general speed guarantee.')
                    (path/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
                    assert report['status']=='PASS_MATCHED_SERVING_SCREEN',report['repeat']
                elif phase=='quality':
                    state['phases'][phase]['commands']=[]
                    (path/'config.json').write_bytes((REPO/'benchmark-results/exl3-optimized-quality-v2/engine-config.json').read_bytes())
                    short=path/'short';short.mkdir();(short/'panel.json').write_bytes((REPO/'benchmark-results/quantization-reference-20261001/panel.json').read_bytes());(short/'bf16').symlink_to('/reference/bf16',target_is_directory=True)
                    long=path/'long';long.mkdir();(long/'panel.json').write_bytes((REPO/'benchmark-results/exl3-long-quality-v3/panel.json').read_bytes())
                    command('short-capture',native_command('/scripts/run_native.py',['--model','/models/checkpoint','--quantization','exl3','--panel','/results/short/panel.json','--out','/results/short/candidate','--engine-config','/results/config.json']))
                    command('short-compare',native_command('/scripts/compare.py',['--root','/results/short','--arms','candidate','--tokenizer','/models/checkpoint/tokenizer.json']))
                    command('long-capture',native_command('/scripts/run_native.py',['--model','/models/checkpoint','--quantization','exl3','--panel','/results/long/panel.json','--out','/results/long/candidate','--engine-config','/results/config.json','--windows','1']))
                    command('quality-check',native_command('/review/check_quality.py',['--root','/results','--existing-short','/v1-short','--long-reference','/long-reference']))
                    replay=path/'replay';replay.mkdir();source=REPO/'benchmarks/experiments/exl3-review-20261002';snapshots=replay/'diagnostic-sources';snapshots.mkdir()
                    for f in ['state_replay.py','run_state_replay.py']:shutil.copyfile(source/f,snapshots/f)
                    (replay/'panel.json.gz').write_bytes(panel_path.read_bytes());(replay/'config.json').write_bytes((path/'config.json').read_bytes())
                    cfg=profile(image);cmd=['docker','run','--rm','--name','b70-exl3-review-quality','--network','none','--device','/dev/dri','--memory','14g','--shm-size','4g','-v',str(MODEL)+':/exl3:ro','-v',str(snapshots)+':/scripts:ro','-v',str(replay)+':/results']
                    for k,v in {**cfg['env'],'EXL3_SDPA_CACHE_CAPACITY':'64','HF_HUB_OFFLINE':'1','PYTHONPATH':'/scripts','ZE_AFFINITY_MASK':'0','OMP_NUM_THREADS':'4','B70_REPLAY_IMAGE_SCOPE':'Final reviewed release candidate; fresh long-prefill and real C4 graph numerical smoke'}.items():cmd+=['-e',k+'='+v]
                    cmd+=['--entrypoint','python',image,'-u','/scripts/run_state_replay.py','--panel','/results/panel.json.gz','--config','/results/config.json','--out','/results/observations']
                    command('candidate-matched-state',cmd)
                    assert read(replay/'observations/summary.json')['status']=='COMPLETE_MATCHED_STATE_DIAGNOSTIC'
                    results=[read(p) for p in (replay/'observations/steps').glob('*-result.json')]
                    assert len(results)==4 and all(d['status']=='PASS_LOCAL_ATTENTION_ORIGINAL_TOLERANCE' for d in results)
                    assert any(d['graph_mode']=='FULL' for d in results) and any(d['label']=='long-prefill' for d in results)
                elif phase=='telemetry':
                    w=worker(path/'serving',env={'B70_REVIEW_CACHE_DIAGNOSTIC':'1','PYTHONPATH':'/opt/b70-review:/opt/b70-runtime'},binds=[(fixture,'/opt/b70-review')])
                    try:
                        w.start();mixed=X.mixed(panel,3,path/'mixed-c4',0)
                        ids=next(x for x in panel['windows'] if x['name']=='code-4096')['ids']
                        lengths=[4352+37*i for i in range(70)];requests=[]
                        for index,length in enumerate(lengths+lengths[-8:]):
                            prompt=(ids*((length+len(ids)-1)//len(ids)))[:length]
                            before=D.R.snapshot(BASE);started=time.monotonic()
                            r=D.R.http(BASE,'/v1/completions',dict(model='Qwen3.8-27B',prompt=prompt,temperature=0,seed=20261001,max_tokens=4,ignore_eos=True,cache_salt=f'review-cache-{index}'),timeout=300)
                            _,native=D.R.wait_accounted(BASE,before,r['usage']);assert native['preemptions']==0 and r['usage']['prompt_tokens']==length
                            requests.append(dict(exact_prompt_tokens=length,wall_s=time.monotonic()-started,usage=r['usage'],native=native,prompt_sha256=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()))
                            (path/'requests.json').write_text(json.dumps(requests,indent=2)+'\n')
                        observations=[json.loads(line) for line in (w.root/'cache-observations.jsonl').read_text().splitlines()]
                        owners={}
                        for d in observations:
                            assert d['cache']['capacity']==64 and d['cache']['entries']<=64 and d['cache']['peak_entries']<=64
                            owners.setdefault((d['pid'],d['thread_id'],d['stream']),[]).append(d)
                        assert any(d['cache']['evictions']>0 and d['cache']['hits']>0 for d in observations)
                        report=dict(status='PASS_ACTUAL_WORKER_CACHE_DIAGNOSTIC',mixed=mixed,requests=len(requests),varied_exact_lengths=len(lengths),observations=len(observations),
                            owners=[dict(pid=k[0],thread_id=k[1],stream=k[2],first=v[0],last=v[-1],rss_min_bytes=min(d['rss_bytes'] for d in v),rss_max_bytes=max(d['rss_bytes'] for d in v)) for k,v in owners.items()],
                            scope='Same inference worker/thread/queue host cache counters; instrumented TTFT/gaps and memory, excluded from serving throughput. Per-context cap, not a global allocator or oneDNN internal-cache limit.')
                        (path/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
                    finally:w.stop()
                state['phases'][phase]['status']='PASS';save();print('PASS',phase,flush=True)
            state['status']='PASS_BOUNDED_REVIEW_RELEASE_QUALIFICATION';success=True
        except BaseException as exc:
            state['status']='FAILED';state['error']=repr(exc);raise
        finally:
            subprocess.run(['docker','rm','-f','b70-exl3-review-quality'],capture_output=True,timeout=60)
            if not success or not a.leave_offline:state['restoration']=S.restore(restore_image)
            state['production_left_offline']=success and a.leave_offline;state['finished_unix']=time.time();save()


if __name__=='__main__':main()
