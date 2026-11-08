#!/usr/bin/env python3
"""Exercise the archived 0.31 launcher exclusively, then restore the service."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from zoneinfo import ZoneInfo

REPO=Path(__file__).resolve().parents[3]
SERVICE='b70-qwen38-vllm.service'
NAME='b70-v033-rollback-smoke'
BASE='http://127.0.0.1:8082'


def http(base,path,body=None):
    request=urllib.request.Request(base+path,
        None if body is None else json.dumps(body).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)


def ready(base,process=None):
    end=time.monotonic()+900
    while time.monotonic()<end:
        if process is not None and process.poll() is not None:
            raise RuntimeError('Strict rollback worker exited; inspect launcher.log')
        try:return http(base,'/v1/models')
        except OSError:time.sleep(2)
    raise TimeoutError('Readiness: '+base)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if dt.datetime.now(ZoneInfo('Europe/Berlin')).hour>=22:
        raise SystemExit('Start a bounded rollback check before22:00')
    assert not (REPO/'.env').exists(), 'Use the review worktree without a serving .env'
    archive=REPO/'config/releases/exl3-v4'
    release=json.loads((archive/'production_image.json').read_text())
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    state={'status':'RUNNING','rollback_image_id':release['image_id']}
    def save():(root/'rollback.json').write_text(json.dumps(state,indent=2)+'\n')
    process=None
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,lambda n,f:(_ for _ in ()).throw(KeyboardInterrupt(str(n))))
    with Path('/tmp/b70-v033-qualification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm'],text=True))[0]
        metrics=urllib.request.urlopen('http://127.0.0.1:8081/metrics',timeout=10).read().decode()
        for line in metrics.splitlines():
            if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')):
                assert float(line.split()[-1])==0, 'Production engine is busy'
        state['production_image_before']=before['Image'];save()
        subprocess.run(['systemctl','--user','stop',SERVICE],check=True,timeout=120)
        try:
            assert subprocess.run(['docker','inspect',NAME],capture_output=True).returncode!=0
            env={**os.environ,'PORT':'8082','VLLM_CONTAINER':NAME}
            with (root/'launcher.log').open('w') as log:
                process=subprocess.Popen(['bash',str(REPO/'scripts/run-server-exl3-v4-rollback.sh')],
                    cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['models']=ready(BASE,process)
                actual=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0]
                assert actual['Image']==release['image_id']
                mounts={x['Destination']:x['Source'] for x in actual['Mounts']}
                assert mounts['/opt/b70-runtime']==str(archive/'runtime')
                variables=dict(x.split('=',1) for x in actual['Config']['Env'])
                assert variables['VLLM_KV_CACHE_LAYOUT']=='LBNHC'
                assert variables['EXL3_TARGET_RUNTIME']=='vllm031'
                reply=http(BASE,'/v1/chat/completions',{
                    'model':'Qwen3.8-27B','messages':[{'role':'user','content':'Name the capital of France. Reply with one word.'}],
                    'max_tokens':64,'temperature':0,'seed':20261108,'reasoning_effort':'none'})
                assert 'paris' in reply['choices'][0]['message']['content'].lower()
                state.update(status='PASS_ARCHIVED_RUNTIME_STRICT_LAUNCH_AND_REAL_DECODE',
                    reply=reply, runtime_mount=mounts['/opt/b70-runtime'],
                    kv_layout=variables['VLLM_KV_CACHE_LAYOUT'])
                save()
        except BaseException as exc:
            state.update(status='FAILED',error=repr(exc));save();raise
        finally:
            if process is not None:
                subprocess.run(['docker','stop','--time','90',NAME],capture_output=True,timeout=105)
                try:process.wait(timeout=120)
                except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=30)
            subprocess.run(['systemctl','--user','start',SERVICE],check=True,timeout=120)
            ready('http://127.0.0.1:8081')
            restored=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm'],text=True))[0]
            assert restored['Image']==before['Image']
            state['production_restored']=True
            state['finished_utc']=dt.datetime.now(dt.UTC).isoformat();save()
    print(state['status'])


if __name__=='__main__':main()
