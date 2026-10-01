#!/usr/bin/env python3
"""Final frozen70-wave matrix, QueueKit v2 and equal-budget Flappy v7 pair.

Measurements do not change production configuration or approve deployment.
The GPTQ service starts only for its paired task and rollback exercise.
"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from exl3_candidate_worker import BASE, REPO, Worker, sha


def module(name,file):
    s=importlib.util.spec_from_file_location(name,REPO/'scripts'/file)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-receipt',type=Path,required=True)
    p.add_argument('--quality-review',type=Path,required=True)
    p.add_argument('--operations-gate',type=Path,required=True)
    p.add_argument('--performance-gate',type=Path,required=True)
    p.add_argument('--fixture-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.out.resolve();assert not root.exists()
    c=module('final_web_campaign','run-web-coding-campaign.py')
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=json.loads(a.image_receipt.read_text());image=receipt['image_id'];runtime=receipt['runtime_image_id']
        assert receipt['rootfs_identical']
        actual=json.loads(subprocess.check_output(['docker','image','inspect',image,runtime],text=True))
        assert actual[0]['RootFS']==actual[1]['RootFS']==receipt['rootfs']
        assert actual[0]['Config']['Labels']['org.local.b70.policy.sha256']==receipt['policy_sha256']
        policy=REPO/'config/experiments/exl3-migration/final-candidate-policy.json'
        assert sha(policy)==receipt['policy_sha256']
        review=json.loads(a.quality_review.read_text())
        assert review['status']=='PASS_FINAL_QUALITY_REVIEW' and review['image_id']==runtime
        for directory,status in [(a.operations_gate,'COMPLETE_FINAL_OPERATIONAL_GATES'),
                                 (a.performance_gate,'COMPLETE_REPEATED_LONG_PERFORMANCE')]:
            gate=json.loads((directory/'campaign.json').read_text())
            assert gate['status']==status and gate['image_id']==runtime
        frozen_path=REPO/'config/experiments/exl3-migration/full-serving-fixture-manifest.json'
        frozen=json.loads(frozen_path.read_text())
        for filename,digest in frozen['files'].items():assert sha(a.fixture_root/filename)==digest,filename
        scenarios=json.loads((REPO/'benchmarks/current-profile-scenarios.json').read_text())
        assert len(scenarios)==20 and sum(x['repeats'] for x in scenarios)==70
        assert sum(x['repeats']*x['concurrency'] for x in scenarios)==124
        root.mkdir(parents=True);state=dict(status='RUNNING',started_unix=time.time(),image_receipt=receipt,
            quality_review_sha256=sha(a.quality_review),fixture_manifest_sha256=sha(frozen_path),
            frozen_fixture_root=str(a.fixture_root.resolve()),sources={f:sha(REPO/'scripts'/f) for f in
              ['run-exl3-final-readme.py','exl3_candidate_worker.py','current-profile-benchmark.py','run-coding-benchmark.py',
               'run-web-coding-benchmark.py','grade-web-coding-output.py','run-server-gptq-rollback.sh']},stages={})
        def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
        def command(name,argv,budget=None):
            state['stages'][name]=dict(status='RUNNING',command=argv);save();print('START',name,flush=True)
            with (root/(name+'.log')).open('w') as log:
                result=subprocess.run(argv,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=budget)
            state['stages'][name]['returncode']=result.returncode;save()
            return result
        save();subprocess.run(['systemctl','--user','stop',c.SERVICE],check=True)
        try:
            for name,argv in [('source-review',[sys.executable,str(REPO/'scripts/current-profile-benchmark.py'),
                 '--base',BASE,'--container','b70-exl3-final-benchmark','--expected-max-num-seqs','16',
                 '--fixture-root',str(a.fixture_root.resolve()),'--legacy-prefix-namespace',
                 '--output-root',str(root/'source-review'),'--execute']),
                ('coding-agent-v2',[sys.executable,str(REPO/'scripts/run-coding-benchmark.py'),
                 '--base',BASE,'--container','b70-exl3-final-benchmark','--policy-sha256-file',
                 str(policy.with_suffix('.sha256')),'--output-root',str(root/'coding-agent-v2')]),
                ('prefix-64k-isolated',[sys.executable,str(REPO/'scripts/current-profile-benchmark.py'),
                 '--base',BASE,'--container','b70-exl3-final-benchmark','--expected-max-num-seqs','16',
                 '--fixture-root',str(a.fixture_root.resolve()),'--legacy-prefix-namespace','--only','prefix-64k-cold-warm',
                 '--output-root',str(root/'prefix-64k-isolated'),'--execute'])]:
                w=Worker(image,root/(name+'-worker'),name='b70-exl3-final-benchmark')
                try:
                    w.start();command(name,argv).check_returncode()
                    state['stages'][name]['status']='COMPLETE';save()
                finally:w.stop()
            flappy=root/'flappy-v7';flappy.mkdir();state['flappy_limits']=dict(stages=6,
                max_requests_per_stage=32,wall_seconds=2400,thinking_token_budget=4096,seed_base=73000);save()
            release=json.loads((REPO/'config/releases/gptq-onednn-v2/production_image.json').read_text())
            assert release['image_id']=='sha256:ed1ebca756abb0e0832d11cd0db026dd7e86df094c6903efe7ae8afbdc290b68'
            for engine in ['gptq','exl3']:
                w=None;rollback_process=None;rollback_log=None
                try:
                    if engine=='gptq':
                        rollback_log=(root/'gptq-rollback-launch.log').open('w')
                        rollback_process=subprocess.Popen(['bash',str(REPO/'scripts/run-server-gptq-rollback.sh')],
                            cwd=REPO,stdout=rollback_log,stderr=subprocess.STDOUT)
                        base='http://127.0.0.1:8081';container='b70-qwen38-vllm'
                        identity=c.wait_health(base,container,release['image_id'])
                        item=json.loads(subprocess.check_output(['docker','inspect',container],text=True))[0]
                        state['rollback_drill']=dict(status='HEALTH_AND_IDENTITY_PASS_TASK_PENDING',identity=identity,
                                                    mounts=item['Mounts'],launch=item['Config']['Cmd']);save()
                    else:
                        base=BASE;container='b70-exl3-final-benchmark'
                        w=Worker(image,flappy/'exl3-worker',name=container);w.start()
                    warm=flappy/(engine+'-warmup');warm.mkdir()
                    models=c.R.http(base,'/v1/models');c.warmup(base,models['data'][0]['id'],warm)
                    result=command('flappy-'+engine,[sys.executable,str(REPO/'scripts/run-web-coding-benchmark.py'),
                        '--fixture',str(REPO/'benchmarks/web-coding-fixture/v7'),'--output',str(flappy/engine),
                        '--base',base,'--container',container,'--stages','6','--max-wall-seconds','2400',
                        '--thinking-token-budget','4096'],4200)
                    measured=json.loads((flappy/engine/'summary.json').read_text())
                    if result.returncode:
                        reason=str(measured.get('error',''))
                        assert measured['status']=='failed' and (reason=='RuntimeError: run wall-time limit'
                            or reason.startswith('RuntimeError: common context window exhausted:')),reason
                    command('grade-flappy-'+engine,[sys.executable,str(REPO/'scripts/grade-web-coding-output.py'),
                        str(flappy/engine)],600).check_returncode()
                    state['stages']['flappy-'+engine].update(status='MEASURED_AND_GRADED',task_status=measured['status'],
                        task_stop_reason=measured.get('error'));save()
                    if engine=='gptq':state['rollback_drill']['status']='PASS_RUNNING_FROZEN_GPTQ_WITH_PAIRED_TASK';save()
                finally:
                    if w:w.stop()
                    if rollback_process is not None:
                        item=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm'],text=True))[0]
                        assert item['Image']==release['image_id'], 'Refuse to stop unexpected rollback worker'
                        subprocess.run(['docker','stop','-t','30','b70-qwen38-vllm'],check=True,timeout=60)
                        rollback_process.wait(timeout=60);rollback_log.close()
                    subprocess.run(['systemctl','--user','stop',c.SERVICE],check=True)
            state['status']='COMPLETE_FINAL_README_MEASUREMENTS_REQUIRES_RELEASE_REVIEW'
        except BaseException as exc:
            state['status']='FAILED';state['error']=repr(exc);raise
        finally:state['finished_unix']=time.time();state['production_left_offline']=True;save()


if __name__=='__main__':main()
