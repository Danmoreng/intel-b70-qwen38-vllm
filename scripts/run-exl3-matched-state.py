#!/usr/bin/env python3
"""Run a bounded numerical diagnostic on the unchanged production image."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import time

REPO = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--full-mixed-state',action='store_true')
    p.add_argument('--locate-native-mixed',action='store_true')
    p.add_argument('--early-targets',action='store_true');a=p.parse_args()
    if a.locate_native_mixed:a.full_mixed_state=True
    out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
    original=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm']))[0]
    image=original['Image']
    assert image == json.loads((REPO/'config/production_image.json').read_text())['image_id']
    previous=REPO.parent/'intel-b70-qwen38-vllm-onednn/benchmark-results/exl3-optimized-quality-v2'
    for source,target in [('performance-panel.json.gz','panel.json.gz'),('engine-config.json','config.json')]:
        (out/target).write_bytes((previous/source).read_bytes())
    source=REPO/'benchmarks/experiments/exl3-review-20261002'
    state=dict(status='RUNNING',image_id=image,started_unix=time.time(),
        source_sha256={str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),source/'state_replay.py',source/'run_state_replay.py']})
    snapshots=out/'diagnostic-sources';snapshots.mkdir()
    for f in [Path(__file__),source/'state_replay.py',source/'run_state_replay.py']:
        shutil.copyfile(f,snapshots/f.name)
    def save():(out/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
    cache=Path.home()/'.cache/exl3xpu/pro-review-state'/image.removeprefix('sha256:')
    checkpoint=Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
    command=['docker','run','--rm','--name','b70-exl3-matched-state','--network','none','--device','/dev/dri',
        '--memory','24g' if a.full_mixed_state or a.early_targets else '14g','--shm-size','4g','-v',str(checkpoint)+':/exl3:ro','-v',str(snapshots)+':/scripts:ro','-v',str(out)+':/results']
    for key,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
        (cache/key).mkdir(parents=True,exist_ok=True);command+=['-v',str(cache/key)+':'+target]
    environment=dict(HF_HUB_OFFLINE='1',PYTHONPATH='/scripts',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        ZE_AFFINITY_MASK='0',OMP_NUM_THREADS='4',VLLM_XPU_ENABLE_XPU_GRAPH='1',EXL3_TARGET_RUNTIME='vllm030',
        EXL3_INT8_PREFILL='1',EXL3_ONEDNN_ATTN='0',EXL3_DRAFT_VOCAB='/opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw/draft_vocab.json',
        EXL3_GUARDED_ATTN='1',EXL3_GUARDED_PREFILL='1',EXL3_M04_LIBRARY='/opt/exl3xpu/m04/m04.so',
        EXL3_M04_LIBRARY_SHA256='eaa18427db27d4fceeca8a17c8a3c6b019b7678f39bf98affc1736b9f2c2d631',
        EXL3_LOADER_REPORT_DIR='/results/loader',B70_ONEDNN_PREFILL='0',B70_ONEDNN_MIXED_ROUTE='0',B70_GPTQ_W4A8_PREFILL='0')
    if a.full_mixed_state:environment['B70_REPLAY_FULL_MIXED_STATE']='1'
    if a.locate_native_mixed:environment['B70_REPLAY_LOCATE_NATIVE_MIXED']='1'
    if a.early_targets:environment['B70_REPLAY_EARLY_TARGETS']='1'
    for key,value in environment.items():command+=['-e',key+'='+value]
    command+=['--entrypoint','python',image,'-u','/scripts/run_state_replay.py','--panel','/results/panel.json.gz',
        '--config','/results/config.json','--out','/results/replay']
    state['command']=command;save()
    try:
        subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
        with (out/'diagnostic.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        assert json.loads((out/'replay/summary.json').read_text())['status']=='COMPLETE_MATCHED_STATE_DIAGNOSTIC'
        state['status']='COMPLETE_MATCHED_STATE_DIAGNOSTIC'
    except BaseException as exc:
        state['status']='FAILED';state['error']=repr(exc);raise
    finally:
        subprocess.run(['docker','rm','-f','b70-exl3-matched-state'],capture_output=True,timeout=30)
        subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
        spec=importlib.util.spec_from_file_location('micro',REPO/'scripts/run-exl3-review-micro.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        state['restoration']=helper.wait_for_qualified_service(image)
        state['finished_unix']=time.time();save()


if __name__=='__main__':main()
