#!/usr/bin/env python3
"""Prepare immutable candidate metadata without changing production config."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
SRC=REPO.parent/'exl3xpu-vllm031'
OUT=REPO/'benchmark-results/vllm-031-upgrade-20261009'
p=argparse.ArgumentParser();p.add_argument('--runtime-image',default='local/b70-qwen38-vllm:v031-exl3-candidate-20261009');p.add_argument('--image-tag',default='local/b70-qwen38-vllm:exl3-release-candidate-v3');p.add_argument('--metadata-dir',default='metadata-image');a=p.parse_args()
CFG=REPO/'config/experiments/vllm-031-upgrade'
CFG.mkdir(parents=True,exist_ok=False)
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((REPO/'config/production_policy.json').read_text())
p['policy_id']='b70-qwen38-exl3-production-v3'
p['runtime'].update(vllm='0.31.0',torch='2.14.0+xpu',native_library_sha256=sha(OUT/'build/native-cxx20/_C.so'),exl3_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SRC,text=True).strip())
p['attention']['m04_library_sha256']=sha(OUT/'build/m04/m04.so')
p['release']['immediate_exl3_rollback_image']=json.loads((REPO/'config/production_image.json').read_text())['image_id']
write(CFG/'production_policy.json',p);h=sha(CFG/'production_policy.json');(CFG/'production_policy.sha256').write_text(h+'  production_policy.json\n')
subprocess.run([sys.executable,str(REPO/'scripts/prepare-exl3-release-image.py'),'--runtime-image',a.runtime_image,'--policy',str(CFG/'production_policy.json'),'--tag',a.image_tag,'--out',str(OUT/a.metadata_dir),'--cache-capacity','64'],check=True)
m=json.loads((OUT/a.metadata_dir/'image.json').read_text())
r=json.loads((REPO/'config/production_image.json').read_text())
r.update(status='CANDIDATE_PREFLIGHT_ONLY',image_tag=m['image_tag'],image_id=m['image_id'],runtime_image_id=m['runtime_image_id'],policy_sha256=h,rootfs_metadata_attestation=m,source_commit=p['runtime']['exl3_source_commit'])
for key in ['qualification_review','production_start_receipt','qualification_note']:r.pop(key,None)
paths=list(r['runtime_artifacts_sha256'])
code="import hashlib,json,pathlib,sys; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in json.loads(sys.argv[1])}))"
r['runtime_artifacts_sha256']=json.loads(subprocess.check_output(['docker','run','--rm','--entrypoint','python',m['image_id'],'-c',code,json.dumps(paths)],text=True))
sys.path.insert(0,str(REPO/'scripts'));from exl3_candidate_worker import profile
r['vllm_config']=profile(m['image_id'])['config']['vllm']
write(CFG/'production_image.json',r)
print(json.dumps(dict(status=r['status'],image_id=r['image_id'],policy_sha256=h)))
