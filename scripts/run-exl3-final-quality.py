#!/usr/bin/env python3
"""Final image quality with existing BF16 references and matched native control.

No original-model inference is repeated. Capture instrumentation bounds only
the measurement head batches to128; this is recorded, not a serving change.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

from exl3_candidate_worker import REPO, MODEL, profile, sha

HELPERS=REPO/'benchmarks/experiments/quantization-reference'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mixed-gate',type=Path,required=True);p.add_argument('--rows-gate',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=REPO/'benchmark-results/quantization-reference-20261001')
    p.add_argument('--long-reference',type=Path,default=REPO/'benchmark-results/exl3-long-quality-v3')
    p.add_argument('--performance-panel',type=Path,default=REPO/'benchmarks/results/exl3-migration/target-mtp-protocol-v2/mtp-panel.json.gz')
    args=p.parse_args();root=args.out.resolve();assert not root.exists()
    image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        mixed=json.loads((args.mixed_gate/'campaign.json').read_text())
        rows=json.loads((args.rows_gate/'probe/result.json').read_text())
        assert mixed['status']=='COMPLETE_MIXED_AND_API_MEDIA_GATES' and mixed['image_id']==image
        assert rows['status']=='PASS_EXPANDED_ROW_PADDING_GRAPH_GATE'
        assert sha(args.reference/'panel.json')=='cbf1a71bbda470859f2c0786cb7134e260111d2072c4753a6732021dadae6171'
        assert sha(args.reference/'reference-bf16.npz')=='41dd8ce312ae11832360649657b936d8a937e8c33bfe6e1382eba083897ef884'
        assert sha(args.long_reference/'panel.json')=='fbb3be674583f1704573d19c55fe5a86187efd35bc4438df2ef100132efeba4d'
        native_sha=subprocess.check_output(['docker','run','--rm','--entrypoint','sha256sum',image,
                                           '/opt/exl3xpu/exl3xpu/_C.so'],text=True).split()[0]
        assert rows['native_library_sha256']==native_sha
        root.mkdir();cfg=profile(image)
        spec=importlib.util.spec_from_file_location('final_quality_stages',REPO/'scripts/run-exl3-quality-stages.py')
        stages=importlib.util.module_from_spec(spec);spec.loader.exec_module(stages)
        engine,_=stages.stage_config('target-full-candidate')
        for key,value in engine.items():
            if key=='enforce_eager':continue
            assert cfg['config']['vllm'].get(key)==value,(key,value)
        (root/'engine-config.json').write_text(json.dumps(engine,indent=2)+'\n')
        state=dict(status='RUNNING',image_id=image,started_unix=time.time(),stages={},
                   engine_config=engine,profile=cfg,reference_reused=True,
                   mixed_campaign_sha256=sha(args.mixed_gate/'campaign.json'),
                   rows_gate_sha256=sha(args.rows_gate/'probe/result.json'),
                   source_sha256={str(x.relative_to(REPO)):sha(x) for x in [Path(__file__),*sorted(HELPERS.glob('*.py'))]})
        def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
        save()
        subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True)
        bf16=(args.reference/'bf16').resolve();long_bf16=(args.long_reference/'bf16').resolve()
        common=['docker','run','--rm','--device','/dev/dri','--memory','12g','--shm-size','4g',
                '-v',str(MODEL)+':/models/checkpoint:ro','-v',str(HELPERS)+':/scripts:ro',
                '-v',str(root)+':/results','-v',str(bf16)+':'+str(bf16)+':ro',
                '-v',str(args.reference.resolve())+':'+str(args.reference.resolve())+':ro',
                '-v',str(long_bf16)+':'+str(long_bf16)+':ro']
        cache=Path.home()/'.cache/exl3xpu/optimized-quality'/image.removeprefix('sha256:')
        for key,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
            (cache/key).mkdir(parents=True,exist_ok=True);common+=['-v',str(cache/key)+':'+target]
        env={**cfg['env'],'HF_HUB_OFFLINE':'1','PYTHONPATH':'/scripts','ZE_AFFINITY_MASK':'0','OMP_NUM_THREADS':'4'}
        def run(stage,script,arguments,optimized=True):
            current={**env,'EXL3_GUARDED_ATTN':'1' if optimized else '0',
                     'EXL3_LOADER_REPORT_DIR':'/results/loader-'+stage}
            command=list(common)
            for k,v in current.items():command+=['-e',k+'='+v]
            command+=['--entrypoint','python',image,'-u','/scripts/'+script,*arguments]
            state['stages'][stage]=dict(status='RUNNING',command=command);save()
            print('START',stage,flush=True)
            with (root/(stage+'.log')).open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            state['stages'][stage]['status']='COMPLETE';save();print('COMPLETE',stage,flush=True)
        try:
            run('reference-integrity','validate_reference_bundle.py',
                ['--root',str(args.reference.resolve()),'--output','/results/reference-integrity.json'])
            short=root/'short';short.mkdir();(short/'panel.json').write_bytes((args.reference/'panel.json').read_bytes())
            (short/'bf16').symlink_to(bf16,target_is_directory=True)
            for arm,opt in [('target-native-control',False),('target-optimized',True)]:
                run('short-'+arm,'run_native.py',['--model','/models/checkpoint','--quantization','exl3',
                    '--panel','/results/short/panel.json','--out','/results/short/'+arm,
                    '--engine-config','/results/engine-config.json'],opt)
            run('short-comparison','compare.py',['--root','/results/short','--arms','target-native-control','target-optimized',
                '--tokenizer','/models/checkpoint/tokenizer.json'])
            long=root/'long';long.mkdir();(long/'panel.json').write_bytes((args.long_reference/'panel.json').read_bytes())
            (long/'bf16').symlink_to(long_bf16,target_is_directory=True)
            run('long-optimized','run_native.py',['--model','/models/checkpoint','--quantization','exl3',
                '--panel','/results/long/panel.json','--out','/results/long/target-full-candidate',
                '--engine-config','/results/engine-config.json'])
            run('long-comparison','compare_long_suffix.py',['--root','/results/long','--tokenizer','/models/checkpoint/tokenizer.json'])
            (root/'performance-panel.json.gz').write_bytes(args.performance_panel.read_bytes())
            for arm,opt in [('generated-native',False),('generated-optimized',True)]:
                argv=['--model','/models/checkpoint','--panel','/results/performance-panel.json.gz',
                      '--engine-config','/results/engine-config.json','--out','/results/'+arm]
                if opt:argv+=['--reference','/results/generated-native/summary.json']
                run(arm,'run_generation_probe.py',argv,opt)
            run('brief-components','run_component_profile.py',['--model','/models/checkpoint',
                '--panel','/results/performance-panel.json.gz','--engine-config','/results/engine-config.json',
                '--out','/results/brief-components','--compact','--output-tokens','128'])
            state['status']='COMPLETE_MEASURED_FINAL_QUALITY_REQUIRES_REVIEW'
        except BaseException as exc:state['status']='FAILED';state['error']=repr(exc);raise
        finally:
            state['production_left_offline']=True;state['finished_unix']=time.time();save()


if __name__=='__main__':main()
