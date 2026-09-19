"""Exclusive operator gate, with original production restored on every exit."""
import datetime,fcntl,json,os,signal,subprocess,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
NAME='b70-m10-gdn-operator'
IMAGE='sha256:b675d81d4e7cc63fbcd6df395965ea16ec5c4704428c81118a1618185245dd5a'
def recover(run):
    if not (run/'production-owned').exists():return
    subprocess.run(['docker','stop','-t','20',NAME],capture_output=True,timeout=35)
    subprocess.run(['python3',str(REPO/'scripts/restore-production.py')],check=True,timeout=850,env={**os.environ,'B70_RUN_DIR':str(run)})
    (run/'production-owned').unlink()
if len(sys.argv)>1 and sys.argv[1]=='recover':
    recover(Path(sys.argv[2]));sys.exit()
run=Path(sys.argv[1]);run.mkdir(parents=True,exist_ok=True)
lock=Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prod=json.loads(subprocess.check_output(['docker','inspect','qwen38-vllm-production']))[0];assert prod['Image']==IMAGE
metrics=urllib.request.urlopen('http://127.0.0.1:8081/metrics',timeout=5).read().decode()
for name in ['vllm:num_requests_running{','vllm:num_requests_waiting{']:
    rows=[l for l in metrics.splitlines() if l.startswith(name)];assert rows and all(float(l.split()[-1])==0 for l in rows),'Production busy'
cap=next(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('*/power1_cap'));assert int(cap.read_text())==180000000
(run/'production-owned').touch()
signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
try:
    subprocess.run(['systemctl','--user','stop','qwen38.service'],check=True,timeout=120)
    model='/home/sebastian/.cache/huggingface/hub/models--mikeinnyc--Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16'
    cmd=['docker','run','--rm','--name',NAME,'--network=none','--device','/dev/dri','--group-add',str(Path('/dev/dri/renderD128').stat().st_gid),'-e','ZE_AFFINITY_MASK=0','-e','ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE','-v',f'{ROOT}:/src:ro','-v',f'{run}:/work','-v',f'{model}:/checkpoint:ro','-v',f'{model}/snapshots/a47b0c6f0d756bc394c4cc629d5b0ded1acc7001:/model:ro','--entrypoint','python3',IMAGE,'/src/probe.py']
    # Snapshot symlinks resolve ../../blobs from /model; mount the full cache.
    cmd[cmd.index(f'{model}/snapshots/a47b0c6f0d756bc394c4cc629d5b0ded1acc7001:/model:ro')]=f'{model}/snapshots/a47b0c6f0d756bc394c4cc629d5b0ded1acc7001:/model:ro'
    # Expose the shared blob directory at the symlinks' canonical target.
    at=cmd.index('--entrypoint');cmd[at:at]=['-v',f'{model}/blobs:/blobs:ro']
    (run/'command.json').write_text(json.dumps(cmd,indent=2))
    with (run/'probe.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
finally:recover(run)
