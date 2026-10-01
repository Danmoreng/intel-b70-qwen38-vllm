#!/usr/bin/env python3
"""Strict immutable EXL3 production launch; check-only supports release review."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from exl3_candidate_worker import REPO, MODEL, profile, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--release-dir',type=Path,default=REPO/'config')
    a=p.parse_args();directory=a.release_dir.resolve()
    policy_file=directory/'production_policy.json'
    policy=json.loads(policy_file.read_text());digest=sha(policy_file)
    assert digest==(directory/'production_policy.sha256').read_text().split()[0], 'Production policy hash mismatch'
    release=json.loads((directory/'production_image.json').read_text())
    assert release['policy_sha256']==digest
    assert (release['status']=='QUALIFIED_RELEASE' or
            (a.check_only and release['status']=='CANDIDATE_PREFLIGHT_ONLY')), 'Release not approved'
    assert policy['policy_id']=='b70-qwen38-exl3-production-v1'
    image=release['image_tag']
    inspected=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
    assert inspected['Id']==release['image_id'], 'Production image tag moved'
    assert inspected['Config']['Labels']['org.local.b70.policy.sha256']==digest
    cfg=profile(release['image_id'])
    assert sha(REPO/'runtime/request_defaults.py')==release['request_defaults_sha256'], 'Serving middleware changed'
    required={'VLLM_IMAGE':image,'MODEL_ID':policy['model']['id'],'MODEL_REVISION':policy['model']['revision'],
        'SERVED_MODEL_NAME':policy['model']['served_name'],'CONTEXT_SIZE':'262144','GPU_MEMORY_UTILIZATION':'0.965',
        'MAX_NUM_BATCHED_TOKENS':'4096','MAX_NUM_SEQS':'16','SPECULATIVE_TOKENS':'3',
        'SCHEDULER_WATERMARK':'0.0','PREFIX_CACHING':'1','B70_POWER_LIMIT_W':'180','ZE_AFFINITY_MASK':'0'}
    for key,value in required.items():
        assert os.environ.get(key,value)==value, 'Production profile override: '+key
    for key,value in cfg['env'].items():
        assert os.environ.get(key,value)==value, 'Frozen EXL3 environment override: '+key
    assert cfg['config']['vllm']==release['vllm_config'], 'Frozen serving arguments changed'
    for path,digest_expected in release['checkpoint_files_sha256'].items():
        assert sha(MODEL/path)==digest_expected, 'Checkpoint identity mismatch: '+path
    verify="""import hashlib,json,pathlib,runpy,torch,vllm
runpy.run_path('/opt/exl3xpu/scripts/verify_native_artifact.py')
r=json.loads(__import__('sys').argv[1])
assert torch.__version__=='2.13.0+xpu' and vllm.__version__=='0.30.0'
for p,h in r.items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p
torch.ops.load_library('/opt/exl3xpu/m04/m04.so')
assert torch._C._dispatch_has_kernel_for_dispatch_key('b70_exl3_attention::shared_kv_verify_out','XPU')
"""
    subprocess.run(['docker','run','--rm','--network','none','--entrypoint','python',release['image_id'],
        '-c',verify,json.dumps(release['runtime_artifacts_sha256'])],check=True)
    node=Path(os.environ.get('RENDER_NODE','/dev/dri/renderD128'));assert node.exists()
    caps=list(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('*/power1_cap'))
    assert len(caps)==1 and int(caps[0].read_text())==180000000, 'Card power limit must be180W'
    cache=Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'b70-qwen38-vllm-production'/digest/release['image_id'].removeprefix('sha256:')
    print(json.dumps(dict(status='LAUNCH_PREFLIGHT_PASS',release_status=release['status'],
                         image_id=release['image_id'],policy_sha256=digest,cache=str(cache))),flush=True)
    if a.check_only:return
    env={**cfg['env'],'HF_HUB_OFFLINE':'1','PYTHONPATH':'/opt/b70-runtime','ZE_AFFINITY_MASK':'0',
         'EXL3_LOADER_REPORT_DIR':'/results/loader'}
    results=cache/'reports';results.mkdir(parents=True,exist_ok=True)
    argv=['docker','run','--rm','--name',os.environ.get('VLLM_CONTAINER','b70-qwen38-vllm'),
        '--device','/dev/dri','--group-add',str(node.stat().st_gid),'--shm-size','8g',
        '-p',os.environ.get('VLLM_HOST','127.0.0.1')+':'+os.environ.get('PORT','8081')+':8000',
        '-v',str(MODEL)+':/models/checkpoint:ro','-v',str(REPO/'runtime')+':/opt/b70-runtime:ro',
        '-v',str(results)+':/results']
    for key,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
        (cache/key).mkdir(parents=True,exist_ok=True);argv+=['-v',str(cache/key)+':'+target]
    for key,value in env.items():argv+=['-e',key+'='+value]
    argv+=['--entrypoint','/opt/venv/bin/vllm',release['image_id'],*cfg['argv'][1:]]
    os.execvp(argv[0],argv)


if __name__=='__main__':main()
