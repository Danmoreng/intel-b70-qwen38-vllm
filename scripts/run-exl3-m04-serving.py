#!/usr/bin/env python3
"""Fresh M04 serving candidate with exact reused fixed-MTP3/pruned baseline."""
import argparse
import fcntl
import gzip
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
NAME=M.NAME


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['panel','manifest','build','micro-campaign','baseline','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--image',required=True);args=parser.parse_args();root=args.out.resolve();assert not root.exists()
    panel=json.loads(gzip.decompress(args.panel.read_bytes()));manifest=json.loads(args.manifest.read_text())
    assert sha(args.panel)==manifest['panel_gzip_sha256']
    cases=[(w,c,mode) for w in panel['windows'] for c in [1,4] for mode in ['cold','warm']
           if w['name']=='code-4096' or (w['name']=='code-49152' and c==4) or (w['name']=='code-102752' and c==1)]
    assert len(cases)==8
    build=args.build.resolve();library=json.loads((build/'manifest.json').read_text())
    assert sha(build/'m04.so')==library['library_sha256']
    image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        micro=json.loads((args.micro_campaign/'campaign.json').read_text())
        assert micro['status']=='COMPLETE_MICRO_GATES_ONLY_SERVING_UNQUALIFIED'
        assert micro['image_id']==image and micro['library_sha256']==library['library_sha256']
        prior=json.loads((args.baseline/'campaign.json').read_text())
        assert prior['status']=='COMPLETE' and prior['image_id']==image and prior['depth']==3
        assert prior['panel_sha256']==manifest['panel_raw_sha256']
        baseline=prior['arms'][0];assert baseline['vocabulary']=='pruned' and baseline['status']=='COMPLETE'
        reference=[]
        for window,c,mode in cases:
            key=(window['name'],c,mode)
            matches=[v for v in baseline['waves'] if (v['window'],v['concurrency'],v['cache_mode'])==key];assert len(matches)==1
            path=args.baseline/'pruned'/f'{key[0]}-c{c}-{mode}'/'result.json'
            raw=json.loads(path.read_text());assert {k:v for k,v in raw.items() if k!='responses'}==matches[0]
            assert raw['native']['completed']==c and all(r['prompt_ids_verified'] and len(r['token_ids'])==512 for r in raw['responses'])
            reference.append({**matches[0],'reused_result':str(path.resolve()),'reused_result_sha256':sha(path)})
        root.mkdir();settings={**baseline['settings'],'worker_extension_cls':'install_serving.M04Extension'}
        state={'status':'RUNNING','image_id':image,'started_unix':time.time(),'panel_sha256':manifest['panel_raw_sha256'],
            'settings':settings,'baseline_settings':baseline['settings'],'baseline_waves':reference,'waves':[],
            'library_sha256':library['library_sha256'],'micro_campaign_sha256':sha(args.micro_campaign/'campaign.json'),
            'baseline_campaign_sha256':sha(args.baseline/'campaign.json'),
            'source_sha256':{str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),SOURCE/'install_serving.py',SOURCE/'verify_attention.py',REPO/'scripts/run-exl3-mtp-study.py']},
            'design':'Eight fresh code C1/C4 cold/warm waves against exact reused pruned MTP3 baseline. Same512-output budgets, settings/weights/image; only separate M04 overlay + startup worker extension differ. Single pass; no ABBA/order confidence.',
            'scope':'M04 serving benefit is unproven until this candidate completes. No production change.'}
        def save():M.save(root/'campaign.json',state)
        def interrupted(signum,frame):raise InterruptedError(f'Signal {signum}')
        signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted);save()
        try:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90);M.R.cap()
            checkpoint=Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
            command=['docker','run','-d','--name',NAME,'--device','/dev/dri','--shm-size','8g','-p','127.0.0.1:8082:8000',
                '-e','HF_HUB_OFFLINE=1','-e','PYTHONPATH=/opt/b70-runtime:/opt/m04',
                '-v',str(checkpoint)+':/models/checkpoint:ro','-v',str(REPO/'runtime')+':/opt/b70-runtime:ro',
                '-v',str(SOURCE)+':/opt/m04:ro','-v',str(build)+':/opt/m04-library:ro','-v',str(root)+':/results']
            cache=Path.home()/'.cache/exl3xpu/migration-m04'/image.removeprefix('sha256:')/library['library_sha256']
            for key,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
                (cache/key).mkdir(parents=True,exist_ok=True);command+=['-v',str(cache/key)+':'+target]
            command+=[image,'models/qwen3.8-27b-exl3-4.00bpw/migration-target-200704-c4.yaml','--gpu','0','--port','8000','--model-path','/models/checkpoint']
            for key,value in settings.items():command+=['--set','vllm.'+key+'='+json.dumps(value,separators=(',',':'))]
            command+=['--set','env.EXL3_M04_LIBRARY=/opt/m04-library/m04.so',
                '--set','env.EXL3_M04_LIBRARY_SHA256='+library['library_sha256'],
                '--set','env.EXL3_LOADER_REPORT_DIR=/results/loader-m04']
            state['command']=command;save();subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            deadline=time.monotonic()+900
            while True:
                try:M.R.http(M.BASE,'/v1/models',timeout=3);break
                except OSError:
                    item=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0]
                    assert item['State']['Running'],'M04 engine exited during startup'
                    if time.monotonic()>deadline:raise TimeoutError('M04 startup timeout')
                    time.sleep(2)
            state['identity']=M.R.identity(NAME);assert state['identity']['image_id']==image;save()
            short=next(w for w in panel['windows'] if w['name']=='code-4096')
            for c in [1,4]:M.wave(short,c,'warmup',0,3,root,512)
            for window,c,mode in cases:
                print(f'RUN M04 {window["name"]} C{c} {mode}',flush=True)
                state['waves'].append(M.wave(window,c,mode,0,3,root,512));save()
            state['status']='COMPLETE_SERVING_SCREEN_NOT_RELEASE_QUALIFICATION'
        except BaseException as exc:state['status']='FAILED';state['error']=repr(exc);raise
        finally:
            logs=subprocess.run(['docker','logs',NAME],capture_output=True,text=True)
            (root/'worker.log').write_text(logs.stdout+logs.stderr)
            subprocess.run(['docker','rm','-f',NAME],capture_output=True,timeout=60)
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            state['finished_unix']=time.time();state['production_left_offline']=True;save()


if __name__=='__main__':main()
