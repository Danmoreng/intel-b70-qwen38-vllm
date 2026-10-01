#!/usr/bin/env python3
"""Run EXL3 native gates; leave GPTQ offline unless a rollback drill is requested."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

REPO=Path(__file__).resolve().parents[1]
EXL=REPO.parent/'exl3xpu-migration'
MODEL=Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
IMAGE='sha256:cba73584f4ab0a2b37eac1356f34f16655ac5e740d845e110997b03be78279b7'
SERVICE='b70-qwen38-vllm.service'

def run(cmd,**kw): return subprocess.run(cmd,check=True,**kw)
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def stop_signal(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=REPO/'benchmarks/results/exl3-migration/safe-foundation')
    ap.add_argument('--restore-production',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=a.output/'campaign.json'
    if report.exists(): raise SystemExit('Choose a fresh output directory; evidence is immutable')
    baseline=json.loads((REPO/'config/production_image.json').read_text())['image_id']
    installed=subprocess.check_output(['docker','image','inspect',baseline,'--format','{{.Id}}'],text=True).strip()
    assert installed==baseline
    state={'schema':1,'started_unix':time.time(),'production_image':baseline,'candidate_image':IMAGE,
           'source_sha256':{str(p.relative_to(EXL)):sha(p) for name in ('exl3xpu','tests','scripts')
                            for p in (EXL/name).rglob('*') if p.is_file() and p.suffix in ('.py','.sh')},
           'library_sha256':sha(EXL/'exl3xpu/_C.so'),'steps':[],'production_restored':False,
           'restore_production_requested':a.restore_production}
    def save(): report.write_text(json.dumps(state,indent=2)+'\n')
    for s in (signal.SIGTERM,signal.SIGINT): signal.signal(s,stop_signal)
    save(); stopped=False
    try:
        run(['systemctl','--user','stop',SERVICE]); stopped=True
        for label,script,extra,wanted in (
            ('bitexact-full','test_bitexact_xpu.py',[],0),
            ('int8-prefill','test_int8_prefill.py',['8192'],0),
            ('int8-prefill-injected-failure','test_int8_prefill.py',['8192','--inject-error'],1)):
            name='b70-exl3-gate-'+label
            command=['docker','run','--rm','--name',name,'--memory=14g','--device','/dev/dri:/dev/dri',
                     '-v','/dev/dri/by-path:/dev/dri/by-path:ro',
                     '-v',str(EXL)+':/work:ro','-v',str(MODEL)+':/models/checkpoint:ro',
                     '-v',str(a.output.resolve())+':/results','-w','/work',
                     '-e','PYTHONPATH=/work','-e','MODEL=/models/checkpoint',
                     '-e','EXL3_LIB=/work/exl3xpu/_C.so',
                     '-e','EXL3_TEST_REPORT=/results/'+label+'.json',
                     '--entrypoint','python',IMAGE,'tests/'+script]+extra
            step={'name':label,'command':command,'started_unix':time.time(),'expected_exit':wanted}
            state['steps'].append(step);save()
            print('START '+label,flush=True)
            try:
                with (a.output/(label+'.log')).open('w') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
                step['exit']=result.returncode;step['finished_unix']=time.time()
                step['pass']=result.returncode==wanted
                if not step['pass']: raise RuntimeError(f'{label} exited {result.returncode}, expected {wanted}')
                print('PASS '+label,flush=True);save()
            finally:
                subprocess.run(['docker','rm','-f',name],capture_output=True)
        state['native_gates']='PASS'
    except BaseException as e:
        state['error']=repr(e);save();raise
    finally:
        if stopped and a.restore_production:
            run(['systemctl','--user','start',SERVICE])
            for _ in range(240):
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8081/health',timeout=2) as response:
                        if response.status==200: break
                except (OSError,TimeoutError): pass
                time.sleep(2)
            else: state['restore_error']='Production health did not recover'
            identity=subprocess.run(['docker','inspect','b70-qwen38-vllm','--format','{{.Image}}'],capture_output=True,text=True)
            state['restored_image']=identity.stdout.strip()
            state['production_restored']='restore_error' not in state and state['restored_image']==baseline
        elif stopped:
            run(['systemctl','--user','stop',SERVICE])
            state['production_left_offline']=True
        state['finished_unix']=time.time();save()
        print(json.dumps({'native_gates':state.get('native_gates','FAIL'),
                          'production_restored':state['production_restored']}),flush=True)
    if a.restore_production and not state['production_restored']: raise SystemExit(2)

if __name__=='__main__': main()
