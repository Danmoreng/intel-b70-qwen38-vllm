#!/usr/bin/env python3
"""Measure the frozen reviewed release: serving only, no adaptive coding pair.

Failure restores the saved qualified service. Successful --leave-offline is
for exclusive follow-on release gates; it never promotes the candidate.
"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from exl3_candidate_worker import BASE, REPO, Worker, sha


def loader(root):
    reports=list((root/'loader').glob('loader-*.json'))
    assert len(reports)==1, reports
    d=json.loads(reports[0].read_text())
    assert d['status']=='PASS' and d['loaded_modules']==d['expected_modules']==409
    assert d['mtp_loaded']==d['mtp_expected']==8 and not d['missing'] and not d['unexpected_duplicates']
    return dict(path=str(reports[0].relative_to(REPO)),sha256=sha(reports[0]),loaded_modules=409,mtp_loaded=8)


def restore(image):
    subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
    s=importlib.util.spec_from_file_location('review_recovery',REPO/'scripts/run-exl3-review-micro.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.wait_for_qualified_service(image)


def current_qualified_image():
    from release_integrity import load_release
    return load_release()['image_id']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-dir',type=Path,required=True)
    p.add_argument('--image-receipt',type=Path,required=True)
    p.add_argument('--fixture-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--leave-offline',action='store_true')
    a=p.parse_args(); root=a.out.resolve(); assert not root.exists()
    directory=a.release_dir.resolve(); receipt=json.loads(a.image_receipt.read_text())
    release=json.loads((directory/'production_image.json').read_text())
    saved=json.loads((REPO/'config/releases/exl3-v1/production_image.json').read_text())
    assert release['image_id']==receipt['image_id'] and release['status']=='CANDIDATE_PREFLIGHT_ONLY'
    assert receipt['runtime_image_id']=='sha256:0e711fea1f9231a25289d812fffbde51ed93cbe7bad16c34f7fde3edf3d91737'
    items=json.loads(subprocess.check_output(['docker','image','inspect',receipt['image_id'],receipt['runtime_image_id']],text=True))
    assert items[0]['RootFS']==items[1]['RootFS']==receipt['rootfs']
    assert items[0]['Config']['Labels']['org.local.b70.policy.sha256']==sha(directory/'production_policy.json')==receipt['policy_sha256']
    frozen=REPO/'config/experiments/exl3-migration/full-serving-fixture-manifest.json'
    fixture=json.loads(frozen.read_text())
    for name,digest in fixture['files'].items():assert sha(a.fixture_root/name)==digest,name
    plan=json.loads((REPO/'benchmarks/current-profile-scenarios.json').read_text())
    assert len(plan)==20 and sum(x['repeats'] for x in plan)==70 and sum(x['repeats']*x['concurrency'] for x in plan)==124
    caps=list(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('*/power1_cap'))
    assert len(caps)==1 and int(caps[0].read_text())==180000000
    # These checks use CPU-only disposable containers, leaving the serving GPU alone.
    subprocess.run([sys.executable,str(REPO/'scripts/run-server-exl3.py'),'--check-only','--release-dir',str(directory)],check=True)
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        restore_image=current_qualified_image()
        root.mkdir(parents=True)
        state=dict(status='RUNNING',started_unix=time.time(),image_receipt=receipt,
            policy_path=str((directory/'production_policy.json').relative_to(REPO)),
            fixture_manifest_sha256=sha(frozen),frozen_fixture_root=str(a.fixture_root.resolve()),
            effective_partition_cache_capacity=64,power_cap_w=180,restore_image_id=restore_image,
            sources={f:sha(REPO/'scripts'/f) for f in ['run-exl3-review-serving.py','exl3_candidate_worker.py','current-profile-benchmark.py']},stages={})
        def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
        save(); success=False
        subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
        try:
            others=json.loads(subprocess.check_output(['docker','ps','--format','{{json .}}'],text=True).strip().replace('\n',',').join(['[',']']))
            assert not any('b70-exl3' in x['Names'] or x['Names']=='b70-qwen38-vllm' for x in others),others
            for name,only in [('source-review',None),('prefix-64k-isolated','prefix-64k-cold-warm')]:
                w=Worker(receipt['image_id'],root/(name+'-worker'),name='b70-exl3-review-benchmark',env={'EXL3_SDPA_CACHE_CAPACITY':'64'})
                command=[sys.executable,str(REPO/'scripts/current-profile-benchmark.py'),'--base',BASE,
                    '--container',w.name,'--expected-max-num-seqs','16','--fixture-root',str(a.fixture_root.resolve()),
                    '--legacy-prefix-namespace','--output-root',str(root/name),'--execute']
                if only:command+=['--only',only]
                state['stages'][name]=dict(status='RUNNING',command=command);save()
                print('START',name,receipt['image_id'],flush=True)
                try:
                    w.start();state['stages'][name]['loader']=loader(w.root);save()
                    with (root/(name+'.log')).open('w') as log:
                        subprocess.run(command,check=True,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                    state['stages'][name]['status']='COMPLETE';save()
                finally:w.stop()
                print('COMPLETE',name,flush=True)
            state['status']='COMPLETE_REVIEW_SERVING_MATRIX';success=True
        except BaseException as exc:
            state['status']='FAILED';state['error']=repr(exc);raise
        finally:
            if not success or not a.leave_offline:state['restoration']=restore(restore_image)
            state['production_left_offline']=success and a.leave_offline
            state['finished_unix']=time.time();save()


if __name__=='__main__':main()
