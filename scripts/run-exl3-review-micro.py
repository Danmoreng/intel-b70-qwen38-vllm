#!/usr/bin/env python3
"""Focused review XPU gates; always restore the untouched qualified service."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

REPO=Path(__file__).resolve().parents[1]

def wait_for_qualified_service(image, timeout=900):
    """Type=exec returns before preflight and model loading have finished."""
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            current = json.loads(subprocess.check_output(
                ['docker', 'inspect', 'b70-qwen38-vllm'], stderr=subprocess.DEVNULL))[0]
            if current['Image'] != image:
                raise RuntimeError('Restored service has a different image')
            with urllib.request.urlopen('http://127.0.0.1:8081/v1/models', timeout=3) as response:
                models = json.load(response)
            assert any(m['id'] == 'Qwen3.8-27B' for m in models['data'])
            return {'image_id': current['Image'], 'policy_sha256': current['Config']['Labels']['org.local.b70.policy.sha256'],
                    'models_endpoint_healthy': True, 'verified_unix': time.time()}
        except (subprocess.CalledProcessError, OSError, ValueError, AssertionError) as exc:
            last_error = repr(exc)
            time.sleep(2)
    raise RuntimeError(f'Qualified service did not become healthy: {last_error}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--m04-q4-repeat',action='store_true')
    p.add_argument('--m04-copy-only',action='store_true');a=p.parse_args()
    out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
    image=subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
    original=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm']))[0]
    baseline=subprocess.check_output(['docker','exec','b70-qwen38-vllm','python','-c',"from pathlib import Path; print(Path('/opt/exl3xpu/exl3xpu/shared_kv_verify.py').read_text(),end='')"])
    (out/'qualified_shared_kv_verify.py').write_bytes(baseline)
    source=REPO/'benchmarks/experiments/exl3-review-20261002'
    state=dict(status='RUNNING',image_id=image,original_image_id=original['Image'],started_unix=time.time(),stages={},
        baseline_copy_sha256=hashlib.sha256(baseline).hexdigest(),sources_sha256={str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),*source.glob('*.py')]})
    def save():(out/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
    def run(name,arguments,env=None):
        cmd=['docker','run','--rm','--network','none','--device','/dev/dri','--memory','12g','--shm-size','4g',
             '-v',str(out)+':/results','-v',str(source)+':/review:ro','-e','PYTHONPATH=/opt/exl3xpu','--entrypoint','python']
        for k,v in (env or {}).items():cmd+=['-e',k+'='+str(v)]
        cmd +=[image,'-u',*arguments];state['stages'][name]=dict(status='RUNNING',command=cmd);save()
        with (out/(name+'.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        state['stages'][name]['status']='PASS';save()
    save()
    try:
        subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
        if not a.m04_q4_repeat and not a.m04_copy_only:
            run('guarded-attention',['/opt/exl3xpu/tests/test_guarded_attention_xpu.py','--library','/opt/exl3xpu/m04/m04.so','--out','/results/guarded-attention'])
            run('cache-cap8',['/review/cache_soak.py','--out','/results/cache-cap8.json'],{'EXL3_SDPA_CACHE_CAPACITY':8})
            run('cache-default64',['/review/cache_soak.py','--out','/results/cache-default64.json'])
        extra=['--rows','4','--rounds','20','--iterations','100'] if a.m04_q4_repeat else []
        run('m04-copy',['/review/m04_copy_xpu.py','--baseline','/results/qualified_shared_kv_verify.py','--out','/results/m04-copy.json',*extra])
        state['status']='COMPLETE_FOCUSED_XPU_GATES_NOT_PRODUCTION_PROMOTION'
    except BaseException as exc:state['status']='FAILED';state['error']=repr(exc);raise
    finally:
        subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=900)
        state['restoration'] = wait_for_qualified_service(original['Image'])
        state['qualified_service_restored_same_image']=True
        state['finished_unix']=time.time();save()

if __name__=='__main__':main()
