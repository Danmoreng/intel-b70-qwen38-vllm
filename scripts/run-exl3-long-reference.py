#!/usr/bin/env python3
"""Run a bounded-memory original-model suffix reference on the exclusive B70."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'benchmarks/experiments/quantization-reference'
ORIGINAL = 'hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
NAME = 'b70-exl3-long-reference'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--root', type=Path, required=True, help='Prepared panel.json directory')
    parser.add_argument('--qual-campaign', type=Path, required=True)
    parser.add_argument('--chunk-tokens', type=int, default=4096)
    parser.add_argument('--windows', type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    panel = json.loads((root / 'panel.json').read_text())
    if panel['source_panel_sha256'] != 'cbf1a71bbda470859f2c0786cb7134e260111d2072c4753a6732021dadae6171':
        raise RuntimeError('Unknown frozen input panel')
    if (root / 'reference-campaign.json').exists() or (root / 'bf16').exists():
        raise RuntimeError('Refusing to overwrite reference evidence')
    image = subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    hf = Path.home() / '.cache/huggingface'
    if not (hf / ORIGINAL / 'config.json').is_file():
        raise RuntimeError('Pinned original snapshot is missing')
    lockpath = REPO.parent / 'Local-AI-B70/qwen38/context-benchmark/run.lock'
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        contract = json.loads((args.qual_campaign / 'campaign.json').read_text())
        if contract['status'] != 'COMPLETE' or contract['image'] != image:
            raise RuntimeError('Candidate runtime qualification must finish first')
        state = {'schema':1,'status':'RUNNING','image_id':image,
                 'original_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
                 'panel_sha256':sha(root / 'panel.json'), 'chunk_tokens':args.chunk_tokens,
                 'qual_campaign_sha256':sha(args.qual_campaign / 'campaign.json'),
                 'source_sha256':{str(p.relative_to(REPO)):sha(p) for p in
                     (Path(__file__),SCRIPTS/'stream_reference.py',SCRIPTS/'stream_long_reference.py')},
                 'started_unix':time.time(), 'production_restored':False}

        def save():
            (root / 'reference-campaign.json').write_text(json.dumps(state,indent=2)+'\n')

        def interrupted(signum,frame):
            raise InterruptedError(f'Signal {signum}')

        signal.signal(signal.SIGINT,interrupted)
        signal.signal(signal.SIGTERM,interrupted)
        save()
        try:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            cache = Path.home()/'.cache/exl3xpu/migration-long-reference'/image.removeprefix('sha256:')
            cache.mkdir(parents=True,exist_ok=True)
            command = ['docker','run','--rm','--name',NAME,'--device','/dev/dri',
                '-v','/dev/dri/by-path:/dev/dri/by-path:ro','--network','none','--memory','14g','--shm-size','2g',
                '-v',str(hf)+':/cache:ro','-v',str(SCRIPTS)+':/scripts:ro','-v',str(root)+':/results',
                '-v',str(cache)+':/root/.cache','-e','HF_HUB_OFFLINE=1','-e','ZE_AFFINITY_MASK=0',
                '-e','OMP_NUM_THREADS=4','--entrypoint','python',image,'-u','/scripts/stream_long_reference.py',
                '--model','/cache/'+ORIGINAL,'--panel','/results/panel.json','--out','/results/bf16',
                '--chunk-tokens',str(args.chunk_tokens)]
            if args.windows:
                command += ['--windows',str(args.windows)]
            state['command']=command
            save()
            print(json.dumps({'command':command}),flush=True)
            with (root / 'reference.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            state['status']='COMPLETE'
            state['summary_sha256']=sha(root / 'bf16/summary.json')
        except BaseException as exc:
            state['status']='FAILED'
            state['error']=repr(exc)
            raise
        finally:
            subprocess.run(['docker','rm','-f',NAME],capture_output=True,timeout=30)
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            state['production_left_offline']=True
            state['finished_unix']=time.time()
            save()
            print(json.dumps({'status':state['status'],'root':str(root)}),flush=True)


if __name__ == '__main__':
    main()
