"""Exclusive first operator screen; recover original service on every exit."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
NAME='b70-e01-operator'
BASE='sha256:b675d81d4e7cc63fbcd6df395965ea16ec5c4704428c81118a1618185245dd5a'
CANDIDATE='local/qwen38-b70-vllm:e01-qk-rope-gate-20260919'

def recover(run):
    if not (run/'production-owned').exists():return
    subprocess.run(['docker','stop','-t','20',NAME],capture_output=True,timeout=35)
    subprocess.run(['python3',str(REPO/'scripts/restore-production.py')],check=True,timeout=850,
                   env={**os.environ,'B70_RUN_DIR':str(run)})
    (run/'production-owned').unlink()

if sys.argv[1]=='recover':
    recover(Path(sys.argv[2]));sys.exit()
run=Path(sys.argv[1]);run.mkdir(parents=True,exist_ok=True)
lock=Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prod=json.loads(subprocess.check_output(['docker','inspect','qwen38-vllm-production']))[0]
assert prod['Image']==BASE
metrics=urllib.request.urlopen('http://127.0.0.1:8081/metrics',timeout=5).read().decode()
for name in ['vllm:num_requests_running{','vllm:num_requests_waiting{']:
    rows=[l for l in metrics.splitlines() if l.startswith(name)]
    assert rows and all(float(l.split()[-1])==0 for l in rows),'Production busy'
cap=next(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('*/power1_cap'))
assert int(cap.read_text())==180000000
candidate=json.loads(subprocess.check_output(['docker','image','inspect',CANDIDATE]))[0]['Id']
(run/'manifest.json').write_text(json.dumps({'baseline':BASE,'candidate':candidate,
    'production_arguments':prod['Args'],'power_w':180,'source_hashes':{
        p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file()},
    'gate':{'qk_atol':.01,'qk_rtol':.002,'gate_v_exact':True,'graph_latency_reduction_pct':2},
    'scope':'first screen: synthetic QKV + actual checkpoint norm weights; no serving qualification'},indent=2))
(run/'production-owned').touch()
signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
try:
    subprocess.run(['systemctl','--user','stop','qwen38.service'],check=True,timeout=120)
    model=Path.home()/'.cache/huggingface/hub/models--mikeinnyc--Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16'
    snapshot=model/'snapshots/a47b0c6f0d756bc394c4cc629d5b0ded1acc7001'
    cmd=['docker','run','--rm','--name',NAME,'--network=none','--device','/dev/dri',
        '--group-add',str(Path('/dev/dri/renderD128').stat().st_gid),
        '-e','ZE_AFFINITY_MASK=0','-e','ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE',
        '-e','B70_FUSED_QK_ROPE_GATE=1','-e','VLLM_TARGET_DEVICE=xpu',
        '-v',f'{ROOT}:/src:ro','-v',f'{run}:/work','-v',f'{snapshot}:/model:ro',
        '-v',f'{model}/blobs:/blobs:ro','--entrypoint','python',candidate,'/src/probe.py']
    (run/'command.json').write_text(json.dumps(cmd,indent=2))
    with (run/'probe.log').open('w') as log:
        subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
finally:
    recover(run)
