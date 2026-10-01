#!/usr/bin/env python3
"""Score the qualified full EXL3 precision path against frozen long BF16 suffixes."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
SCRIPTS=REPO/'benchmarks/experiments/quantization-reference'
NAME='b70-exl3-long-candidate'
ARM='target-full-candidate'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--quality-campaign',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve();quality=args.quality_campaign.resolve()
    if (root/ARM).exists() or (root/'candidate-campaign.json').exists():
        raise RuntimeError('Fresh candidate evidence required')
    image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    checkpoint=Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
    lockpath=REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock'
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        q=json.loads((quality/'campaign.json').read_text())
        old=q['stages'][ARM]
        if q['status']!='COMPLETE' or q['image_id']!=image or old['status']!='COMPLETE':
            raise RuntimeError('Full-candidate short-panel precision must finish first')
        if sha(quality/ARM/'summary.json')!=old['summary_sha256']:
            raise RuntimeError('Full precision source evidence changed')
        ref=json.loads((root/'reference-campaign.json').read_text())
        if (ref['status']!='COMPLETE' or ref['image_id']!=image
                or sha(root/'bf16/summary.json')!=ref['summary_sha256']
                or sha(root/'panel.json')!=ref['panel_sha256']):
            raise RuntimeError('Complete matched long BF16 reference required')
        config=old['engine_config']
        panel=json.loads((root/'panel.json').read_text())
        if max(len(w['ids'])+1 for w in panel['windows'])>config['max_model_len']:
            raise RuntimeError('Long panel exceeds qualified candidate context')
        (root/'candidate-config.json').write_text(json.dumps(config,indent=2)+'\n')
        state={'schema':1,'status':'RUNNING','image_id':image,'started_unix':time.time(),
               'panel_sha256':sha(root/'panel.json'),'engine_config':config,
               'quality_campaign_sha256':sha(quality/'campaign.json'),
               'reference_campaign_sha256':sha(root/'reference-campaign.json'),
               'source_sha256':{str(p.relative_to(REPO)):sha(p) for p in
                    [Path(__file__),SCRIPTS/'native_capture.py',SCRIPTS/'run_native.py',SCRIPTS/'compare_long_suffix.py']},
               'production_restored':False}
        def save():
            (root/'candidate-campaign.json').write_text(json.dumps(state,indent=2)+'\n')
        def interrupted(signum,frame):
            raise InterruptedError(f'Signal {signum}')
        signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
        save()
        try:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            cache=Path.home()/'.cache/exl3xpu/migration-long-candidate'/image.removeprefix('sha256:')
            for name in ['vllm','triton','neo_compiler_cache']:(cache/name).mkdir(parents=True,exist_ok=True)
            env={'HF_HUB_OFFLINE':'1','PYTHONPATH':'/scripts','VLLM_WORKER_MULTIPROC_METHOD':'spawn',
                 'ZE_FLAT_DEVICE_HIERARCHY':'COMPOSITE','ZE_AFFINITY_MASK':'0','OMP_NUM_THREADS':'4',
                 'EXL3_TARGET_RUNTIME':'vllm030','EXL3_INT8_PREFILL':str(int(old['int8_prefill'])),
                 'EXL3_ONEDNN_ATTN':'0','EXL3_DRAFT_VOCAB':'/opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw/draft_vocab.json',
                 'EXL3_LOADER_REPORT_DIR':'/results/loader-'+ARM,'VLLM_XPU_ENABLE_XPU_GRAPH':'1',
                 'B70_ONEDNN_PREFILL':'0','B70_ONEDNN_MIXED_ROUTE':'0','B70_GPTQ_W4A8_PREFILL':'0'}
            command=['docker','run','--rm','--name',NAME,'--device','/dev/dri','--network','bridge',
                     '--memory','12g','--shm-size','4g','-v','/dev/dri/by-path:/dev/dri/by-path:ro',
                     '-v',str(checkpoint)+':/exl3:ro','-v',str(SCRIPTS)+':/scripts:ro','-v',str(root)+':/results']
            for name,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
                command+=['-v',str(cache/name)+':'+target]
            for key,value in env.items():command+=['-e',key+'='+value]
            command+=['--entrypoint','python',image,'-u','/scripts/run_native.py','--model','/exl3',
                      '--quantization','exl3','--panel','/results/panel.json','--out','/results/'+ARM,
                      '--engine-config','/results/candidate-config.json']
            state['command']=command;save();print(json.dumps({'command':command}),flush=True)
            with (root/'candidate.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            state['summary_sha256']=sha(root/ARM/'summary.json')
            comparison=['docker','run','--rm','--network','none','--memory','4g',
                        '-v',str(SCRIPTS)+':/scripts:ro','-v',str(root)+':/results',
                        '-v',str(checkpoint/'tokenizer.json')+':/tokenizer.json:ro',
                        '--entrypoint','python',image,'/scripts/compare_long_suffix.py',
                        '--root','/results','--tokenizer','/tokenizer.json']
            with (root/'comparison.log').open('w') as log:
                subprocess.run(comparison,check=True,stdout=log,stderr=subprocess.STDOUT)
            state['comparison_sha256']=sha(root/'long-comparison.json');state['status']='COMPLETE'
        except BaseException as exc:
            state['status']='FAILED';state['error']=repr(exc);raise
        finally:
            subprocess.run(['docker','rm','-f',NAME],capture_output=True,timeout=30)
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            state['production_left_offline']=True;state['finished_unix']=time.time();save()
            print(json.dumps({'status':state['status'],'root':str(root)}),flush=True)


if __name__=='__main__':
    main()
