#!/usr/bin/env python3
"""Freeze release metadata on identical, already tested EXL3 filesystem layers.

This does not approve or deploy a release. The final full benchmark runs on the
resulting immutable image, after its parent runtime passes the remaining gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def inspect(image):return json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-image',required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--tag',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache-capacity',type=int,choices=(64,),help='Freeze the reviewed native partition-cache default explicitly')
    args=p.parse_args();root=args.out.resolve();assert not root.exists()
    assert subprocess.run(['docker','image','inspect',args.tag],capture_output=True).returncode!=0
    base=inspect(args.runtime_image);root.mkdir()
    policy_sha=hashlib.sha256(args.policy.read_bytes()).hexdigest()
    capacity_env=('ENV EXL3_SDPA_CACHE_CAPACITY=64\n' if args.cache_capacity is not None else '')
    dockerfile=f'''FROM {base['Id']}
LABEL org.local.b70.policy.sha256="{policy_sha}" \\
      org.local.b70.exl3.runtime-image="{base['Id']}" \\
      org.opencontainers.image.source="https://github.com/Danmoreng/intel-b70-qwen38-vllm"
ENV B70_POLICY_SHA256={policy_sha}
{capacity_env}\
CMD ["models/qwen3.8-27b-exl3-4.00bpw/migration-target-optimized.yaml", "--gpu", "0"]
'''
    (root/'Dockerfile').write_text(dockerfile)
    with (root/'build.log').open('w') as log:
        subprocess.run(['docker','build','--network','none','-t',args.tag,str(root)],check=True,stdout=log,stderr=subprocess.STDOUT)
    child=inspect(args.tag)
    assert base['RootFS']==child['RootFS'], 'Release metadata must not change runtime filesystem payload'
    assert base['Config']['Entrypoint']==child['Config']['Entrypoint']
    old_env=dict(v.split('=',1) for v in base['Config']['Env']);new_env=dict(v.split('=',1) for v in child['Config']['Env'])
    changes={k:(old_env.get(k),new_env.get(k)) for k in old_env.keys()|new_env.keys() if old_env.get(k)!=new_env.get(k)}
    expected_changes={'B70_POLICY_SHA256':(old_env.get('B70_POLICY_SHA256'),policy_sha)}
    if args.cache_capacity is not None and old_env.get('EXL3_SDPA_CACHE_CAPACITY')!='64':
        expected_changes['EXL3_SDPA_CACHE_CAPACITY']=(old_env.get('EXL3_SDPA_CACHE_CAPACITY'),'64')
    assert changes==expected_changes
    state=dict(status='METADATA_FROZEN_NOT_RELEASE_APPROVED',image_tag=args.tag,image_id=child['Id'],
               runtime_image_id=base['Id'],policy_sha256=policy_sha,rootfs_identical=True,
               rootfs=child['RootFS'],environment_changes=changes,
               effective_partition_cache_capacity=args.cache_capacity,
               note='Only labels, policy identity env, optional explicit reviewed cache capacity and default profile CMD differ. Frozen optimized YAML unchanged; no rebuild or tensor change.')
    (root/'image.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({k:state[k] for k in ('status','image_id','policy_sha256','rootfs_identical')}))


if __name__=='__main__':main()
