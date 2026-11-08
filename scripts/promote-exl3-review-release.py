#!/usr/bin/env python3
"""Deploy only passed reviewed bytes; recover the saved release on failure."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request

REPO=Path(__file__).resolve().parents[1]
SERVICE='b70-qwen38-vllm.service'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def http(path,payload=None):
    request=urllib.request.Request('http://127.0.0.1:8081'+path,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)


def verify(image,policy):
    d=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm'],text=True))[0]
    assert d['Image']==image and d['Config']['Labels']['org.local.b70.policy.sha256']==policy
    env=dict(x.split('=',1) for x in d['Config']['Env'])
    assert env['EXL3_SDPA_CACHE_CAPACITY']=='64' and 'B70_REVIEW_CACHE_DIAGNOSTIC' not in env
    assert 'EXL3_MIGRATION_TRACE' not in env, 'Instrumented worker must not become production'
    assert any(x['id']=='Qwen3.8-27B' for x in http('/v1/models')['data'])
    basic=dict(model='Qwen3.8-27B',messages=[dict(role='user',content='Name the capital of France. Answer with one word.')],temperature=0,seed=20261001,max_tokens=64,reasoning_effort='none')
    reply=http('/v1/chat/completions',basic);assert 'paris' in reply['choices'][0]['message']['content'].lower()
    stream=dict(basic,stream=True,stream_options={'include_usage':True})
    request=urllib.request.Request('http://127.0.0.1:8081/v1/chat/completions',data=json.dumps(stream).encode(),headers={'Content-Type':'application/json'})
    chunks=[];usage=None;done=False
    with urllib.request.urlopen(request,timeout=180) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            if line.strip()==b'data: [DONE]':done=True;break
            event=json.loads(line[6:]);assert not event.get('error'),event
            chunks.extend(c.get('delta',{}).get('content') or '' for c in event.get('choices',[]))
            usage=event.get('usage') or usage
    assert done and usage and 'paris' in ''.join(chunks).lower()
    # Omit output/thinking/reasoning overrides to exercise the real defaults.
    default=http('/v1/chat/completions',dict(model='Qwen3.8-27B',messages=[dict(role='user',content='Calculate 13 times 29 and state the answer briefly.')],temperature=0,seed=20261001))
    message=default['choices'][0]['message'];assert '377' in (message.get('content') or '')
    assert message.get('reasoning') or message.get('reasoning_content')
    # The public receipt needs the frozen runtime settings, not unrelated
    # inherited image variables or possible host/application credentials.
    from exl3_candidate_worker import profile
    recorded_keys=set(profile(image)['env']) | {
        'EXL3_SDPA_CACHE_CAPACITY','VLLM_KV_CACHE_LAYOUT','HF_HUB_OFFLINE',
        'ZE_AFFINITY_MASK','PYTHONPATH'}
    return dict(image_id=image,policy_sha256=policy,container_id=d['Id'],models_endpoint=True,
        real_generated_reply=reply['choices'][0]['message']['content'],streaming_done=True,streaming_usage=usage,
        defaults_response=default,default_contract='16384 completion,8192 thinking budget,medium reasoning; unchanged hashed middleware plus live request with no overrides',
        command=d['Config']['Cmd'],environment={key:env[key] for key in sorted(recorded_keys) if key in env},
        mounts=d['Mounts'],verified_unix=time.time())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-dir',type=Path,required=True);p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rollback-release',type=Path,default=REPO/'config/releases/exl3-v1')
    p.add_argument('--production-tag',default='local/b70-qwen38-vllm:production-exl3-v2')
    p.add_argument('--runtime-dir',type=Path,
                   help='Install the reviewed RAM-overlay modules with this release; restore them on failure')
    a=p.parse_args();out=a.out.resolve()
    assert not (out/'promotion.json').exists(), 'Refuse to replace an existing deployment receipt'
    decision=json.loads(a.decision.read_text());assert decision['status']=='QUALIFIED_FOR_PRODUCTION_PROMOTION'
    for key in ('gate_assessment','benchmark_summary'):
        assert sha(REPO/decision[key]['path'])==decision[key]['sha256']
    gate=json.loads((REPO/decision['gate_assessment']['path']).read_text());assert gate['status']=='PASS_EXACT_IMAGE_RELEASE_GATES'
    if decision.get('performance_review_triggers'):
        reviewed=decision['performance_review']
        assert sha(REPO/reviewed['path'])==reviewed['sha256']
        performance=json.loads((REPO/reviewed['path']).read_text())
        assert performance['status']=='PASS_REVIEWED_PERFORMANCE_TRADEOFF'
        assert performance['image_id']==decision['image_id']
        assert performance['reviewed_triggers']==decision['performance_review_triggers']
        for receipt in performance['receipts'].values():
            assert sha(REPO/receipt['path'])==receipt['sha256']
    candidate=json.loads((a.release_dir/'production_image.json').read_text());image=candidate['image_id'];policy=candidate['policy_sha256']
    assert image==decision['image_id']==gate['image_id'] and policy==decision['policy_sha256']
    assert sha(a.release_dir/'production_policy.json')==policy
    runtime_source = None
    ram_manifest = None
    runtime_names = ('request_defaults.py', 'kv_admission.py', 'xpu_cpu_offload.py')
    if a.runtime_dir:
        from ram_kv_release import load_overlay
        overlay = load_overlay(REPO, candidate, {})
        assert overlay is not None, 'Runtime installation requires an attested RAM overlay'
        runtime_source = a.runtime_dir.resolve()
        assert runtime_source == (REPO / overlay['runtime_directory']).resolve()
        assert sha(runtime_source/'request_defaults.py') == candidate['request_defaults_sha256']
        ram_manifest = json.loads((REPO/candidate['ram_kv_runtime']['path']).read_text())
        ram_manifest.pop('runtime_directory', None)
    saved=a.rollback_release.resolve();old=json.loads((saved/'production_image.json').read_text())
    assert old['image_id']==decision['immediate_rollback_image_id']
    inventory=json.loads((saved/'snapshot-inventory.json').read_text())
    for name,digest in inventory['files_sha256'].items():assert sha(saved/name)==digest
    working=subprocess.check_output(['systemctl','--user','show',SERVICE,'--property=WorkingDirectory','--value'],text=True).strip()
    assert Path(working).resolve()==REPO
    started=subprocess.check_output(['systemctl','--user','show',SERVICE,'--property=ExecStart','--value'],text=True)
    expected=str(REPO/'scripts/run-server.sh')
    assert 'path='+expected+' ; argv[]='+expected+' ;' in started
    tag=a.production_tag
    assert tag.startswith('local/b70-qwen38-vllm:production-exl3-') and tag!=old['image_tag']
    found=subprocess.run(['docker','image','inspect',tag,'--format','{{.Id}}'],capture_output=True,text=True)
    assert found.returncode!=0 or found.stdout.strip()==image, 'Refuse to overwrite a different release alias'
    current=json.loads((REPO/'config/production_image.json').read_text());assert current['image_id']==old['image_id']
    out.mkdir(parents=True,exist_ok=True);state=dict(status='PROMOTION_IN_PROGRESS',started_unix=time.time(),image_id=image,
        policy_sha256=policy,rollback_image_id=old['image_id'],decision_sha256=sha(a.decision),stages={})
    private=Path.home()/'.cache/b70-qwen38-vllm-promotion-backups'/image.removeprefix('sha256:');private.mkdir(parents=True,exist_ok=True);private.chmod(0o700)
    env_path=REPO/'.env';original_env=env_path.read_bytes() if env_path.exists() else None
    if original_env is not None:
        (private/'env.before').write_bytes(original_env);(private/'env.before').chmod(0o600)
    config_names = ['production_policy.json','production_policy.sha256','production_image.json']
    if runtime_source:
        config_names.append('production_ram_kv.json')
    original_config={name:((REPO/'config'/name).read_bytes() if (REPO/'config'/name).exists() else None)
                     for name in config_names}
    original_runtime={name:(REPO/'runtime'/name).read_bytes() for name in runtime_names} if runtime_source else {}
    for name,data in original_config.items():
        if data is not None:(private/('config-'+name)).write_bytes(data)
    for name,data in original_runtime.items():(private/('runtime-'+name)).write_bytes(data)
    def save():write(out/'promotion.json',state)
    spec=importlib.util.spec_from_file_location('production_recovery',REPO/'scripts/run-exl3-review-micro.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    success=False;save()
    try:
        subprocess.run(['systemctl','--user','stop',SERVICE],check=True,timeout=90)
        assert subprocess.run(['docker','inspect','b70-qwen38-vllm'],capture_output=True).returncode!=0
        subprocess.run(['docker','tag',image,tag],check=True)
        candidate.update(status='QUALIFIED_RELEASE',image_tag=tag,
            qualification_review=dict(path=str(a.decision.resolve().relative_to(REPO)),sha256=sha(a.decision)),
            qualification_note=decision['rationale'])
        if runtime_source:
            for name in runtime_names:shutil.copyfile(runtime_source/name,REPO/'runtime'/name)
            write(REPO/'config/production_ram_kv.json',ram_manifest)
            candidate['ram_kv_runtime'] = dict(path='config/production_ram_kv.json',
                                               sha256=sha(REPO/'config/production_ram_kv.json'))
        for name in ['production_policy.json','production_policy.sha256']:shutil.copyfile(a.release_dir/name,REPO/'config'/name)
        write(REPO/'config/production_image.json',candidate)
        text=original_env.decode() if original_env is not None else (REPO/'.env.example').read_text()
        from exl3_candidate_worker import profile
        frozen_env=profile(image)['env']
        for key,value in [('VLLM_IMAGE',tag),('EXL3_SDPA_CACHE_CAPACITY','64'),*frozen_env.items()]:
            text,count=re.subn(r'^(?:export\s+)?'+key+r'=.*$',key+'='+value,text,flags=re.MULTILINE)
            if not count:text+='\n'+key+'='+value+'\n'
        env_path.write_text(text)
        subprocess.run(['bash','-c','set -a; source "$1"; set +a; exec python3 "$2" --check-only','release-preflight',str(env_path),str(REPO/'scripts/run-server-exl3.py')],check=True,timeout=720)
        subprocess.run(['systemctl','--user','start',SERVICE],check=True,timeout=90)
        state['stages']['first_start_readiness']=helper.wait_for_qualified_service(image)
        state['stages']['first_start_api']=verify(image,policy);save()
        # Check the real permanent service as well as the earlier direct worker.
        subprocess.run(['systemctl','--user','restart',SERVICE],check=True,timeout=90)
        state['stages']['restart_readiness']=helper.wait_for_qualified_service(image)
        state['stages']['restart_api']=verify(image,policy)
        assert state['stages']['restart_api']['container_id']!=state['stages']['first_start_api']['container_id'], 'Permanent service restart must create a new container'
        state['status']='PASS_STRICT_PRODUCTION_DEPLOYMENT_AND_RESTART';state['finished_unix']=time.time();save()
        candidate['production_start_receipt']=dict(path=str((out/'promotion.json').relative_to(REPO)),sha256=sha(out/'promotion.json'))
        write(REPO/'config/production_image.json',candidate);success=True
        print(json.dumps(dict(status=state['status'],image_id=image,policy_sha256=policy),indent=2))
    except BaseException as exc:
        state['status']='FAILED_RECOVERY_REQUIRED';state['error']=repr(exc);save();raise
    finally:
        if not success:
            subprocess.run(['systemctl','--user','stop',SERVICE],check=True,timeout=90)
            for name,data in original_config.items():
                path=REPO/'config'/name
                if data is None:path.unlink(missing_ok=True)
                else:path.write_bytes(data)
            for name,data in original_runtime.items():(REPO/'runtime'/name).write_bytes(data)
            if original_env is None:env_path.unlink(missing_ok=True)
            else:env_path.write_bytes(original_env)
            subprocess.run(['systemctl','--user','start',SERVICE],check=True,timeout=90)
            state['restoration']=helper.wait_for_qualified_service(old['image_id']);state['status']='FAILED_SAVED_RELEASE_RECOVERED';state['finished_unix']=time.time();save()


if __name__=='__main__':main()
