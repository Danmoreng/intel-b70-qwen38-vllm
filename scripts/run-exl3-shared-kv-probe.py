#!/usr/bin/env python3
"""Serialize actual M04 micro gates after the vocabulary screen under GPU lock."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'benchmarks/experiments/exl3-shared-kv-verify'
SPEC=importlib.util.spec_from_file_location('mtp',REPO/'scripts/run-exl3-mtp-study.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
NAME='b70-exl3-m04-probe'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True);parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--vocabulary-campaign',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.out.resolve();assert not root.exists()
    build=args.build.resolve();manifest=json.loads((build/'manifest.json').read_text())
    assert manifest['status']=='COMPILE_AND_CPU_IMPORT_PASS_XPU_UNTESTED'
    assert sha(build/'m04.so')==manifest['library_sha256'] and manifest['torch']=='2.13.0+xpu'
    for relative in ['csrc/shared_kv_verification.cpp','patches/shared-kv-verification.patch']:
        assert sha(SOURCE/relative)==manifest['source_sha256'][relative]
    image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        print('Waiting for exclusive GPU lock, then running M04 micro gates',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        prior=json.loads((args.vocabulary_campaign/'campaign.json').read_text())
        assert prior['status']=='COMPLETE' and prior['image_id']==image
        root.mkdir();state={'status':'RUNNING','image_id':image,'started_unix':time.time(),
            'build_manifest_sha256':sha(build/'manifest.json'),'library_sha256':manifest['library_sha256'],
            'vocabulary_campaign_sha256':sha(args.vocabulary_campaign/'campaign.json'),
            'source_sha256':{str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),SOURCE/'probe_xpu.py',SOURCE/'verify_attention.py']},
            'scope':'Standalone correctness and attention micro timing. No model loaded or serving change; KV is read-only. Production gates remain open.'}
        def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
        def interrupted(signum,frame):raise InterruptedError(f'Signal {signum}')
        signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted);save()
        try:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            M.R.cap()
            command=['docker','run','--name',NAME,'--device','/dev/dri','--entrypoint','/opt/venv/bin/python',
                '-e','PYTHONPATH=/experiment','-e','HF_HUB_OFFLINE=1',
                '-v',str(SOURCE)+':/experiment:ro','-v',str(build)+':/library:ro','-v',str(root)+':/results',
                image,'/experiment/probe_xpu.py','--library','/library/m04.so','--out','/results/probe']
            state['command']=command;save()
            with (root/'worker.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            identity=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0]
            assert identity['Image']==image;state['container_state']=identity['State']
            results=json.loads((root/'probe/result.json').read_text())
            assert results['status']=='COMPLETE_MICRO_GATES_ONLY_SERVING_UNQUALIFIED'
            state['result_sha256']=sha(root/'probe/result.json');state['status']='COMPLETE_MICRO_GATES_ONLY_SERVING_UNQUALIFIED'
        except BaseException as exc:state['status']='FAILED';state['error']=repr(exc);raise
        finally:
            subprocess.run(['docker','rm','-f',NAME],capture_output=True,timeout=30)
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            state['finished_unix']=time.time();state['production_left_offline']=True;save()


if __name__=='__main__':main()
