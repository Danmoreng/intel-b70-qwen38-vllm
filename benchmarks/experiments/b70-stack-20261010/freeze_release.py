#!/usr/bin/env python3
"""Prepare immutable candidate metadata without changing production config."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
SRC=REPO.parent/'exl3xpu-vllm031'
OUT=REPO/'benchmark-results/b70-stack-20261010'
p=argparse.ArgumentParser();p.add_argument('--runtime-image',default='sha256:b772d37937d188bc8492e2e796eafc46668290e68d356f648257e99b06967048');p.add_argument('--image-tag',default='local/b70-qwen38-vllm:exl3-onednn3134-candidate-20261010');p.add_argument('--metadata-dir',default='E01-final-metadata');a=p.parse_args()
CFG=REPO/'config/experiments/b70-stack-20261010'
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
baseline=json.loads((Path(__file__).parent/'baseline-manifest.json').read_text())['production']
assert sha(REPO/'config/production_policy.json')==baseline['policy_sha256']
native=json.loads((OUT/'E01-native-build/receipt.json').read_text())
assert sha(OUT/'E01-native-build/_C.so')==native['native']['library_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SRC,text=True).strip()==native['source_commit']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=SRC,text=True)
a.runtime_image=subprocess.check_output(['docker','image','inspect',a.runtime_image,'--format','{{.Id}}'],text=True).strip()
assert a.runtime_image==native['image_id']
assert json.loads((REPO/'config/production_image.json').read_text())['image_id']==baseline['image_id']
CFG.mkdir(parents=True,exist_ok=False)
p=json.loads((REPO/'config/production_policy.json').read_text())
p['policy_id']='b70-qwen38-exl3-production-v3'
p['runtime'].update(vllm='0.31.0',torch='2.14.0+xpu',onednn='3.13.4',native_library_sha256=native['native']['library_sha256'],exl3_source_commit=native['source_commit'])
assert p['attention']['m04_library_sha256']=='03ed092ce98faa5e1da18ed7f07b5b6862949bee2be0e742a0c28ef42ea1b6b3'
p['runtime']['onednn_source_commit']='c531fe1a5cd631b7f60277cdf6675c588cadbc28'
p['runtime']['onednn_library_sha256']=json.loads((OUT/'E01-oneDNN-build/library-receipt.json').read_text())['library_sha256']
p['release']['immediate_exl3_rollback_image']=json.loads((REPO/'config/production_image.json').read_text())['image_id']
requirements=json.loads((REPO/'config/experiments/vllm-031-upgrade/runtime_requirements.json').read_text())
requirements['onednn']='3.13.4'
requirements['maintenance_native_rebuild']=dict(onednn_source_commit=p['runtime']['onednn_source_commit'],onednn_library_sha256=p['runtime']['onednn_library_sha256'],native_library_sha256=p['runtime']['native_library_sha256'],exl3_source_commit=p['runtime']['exl3_source_commit'],scope='Only custom oneDNN and its linked EXL3 extension change. Python distribution requirements, Torch/vLLM, compiler and M04 remain the qualified v0.31 stack. Existing metadata conflicts are inherited, not fixed by bypassing dependency resolution.')
write(CFG/'runtime_requirements.json',requirements)
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
