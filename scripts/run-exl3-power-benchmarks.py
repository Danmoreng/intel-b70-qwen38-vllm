#!/usr/bin/env python3
"""Repeat the frozen v2 serving matrix at selected power caps, then recover 180 W.

Only the hardware power cap changes. Production policy, image, weights,
serving arguments and benchmark prompts remain frozen.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from exl3_candidate_worker import BASE, REPO, Worker, sha
from release_integrity import load_release


def module(name, file):
    spec=importlib.util.spec_from_file_location(name,REPO/'scripts'/file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


S=module('power_serving','run-exl3-review-serving.py')
E=module('power_export','summarize-readme-benchmarks.py')


def write(path, value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def set_cap(watts):
    assert watts in (150,180,230,275)
    subprocess.run([sys.executable,str(REPO/'scripts/set-power-limit.py')],check=True,
                   env={**os.environ,'B70_POWER_LIMIT_W':str(watts)})


class Hardware:
    def __init__(self, directory, watts):
        self.directory=directory;self.watts=watts;self.stop=threading.Event();self.errors=[]
        self.hwmon=next(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('*/power1_cap')).parent
        assert (self.hwmon/'energy1_label').read_text().strip()=='card'
        self.thread=threading.Thread(target=self.observe,daemon=True)

    def observe(self):
        with (self.directory/'hardware-observations.jsonl').open('w') as log:
            while not self.stop.is_set():
                try:
                    row=dict(unix=time.time(),monotonic=time.monotonic(),
                        power_cap_uw=int((self.hwmon/'power1_cap').read_text()),
                        card_energy_uj=int((self.hwmon/'energy1_input').read_text()),
                        package_temperature_c=int((self.hwmon/'temp2_input').read_text())/1000,
                        vram_temperature_c=int((self.hwmon/'temp3_input').read_text())/1000)
                    if row['power_cap_uw']!=self.watts*1_000_000:raise RuntimeError('Hardware cap changed during measurements')
                    log.write(json.dumps(row)+'\n');log.flush()
                except BaseException as exc:
                    self.errors.append(repr(exc));break
                self.stop.wait(2)

    def start(self):self.thread.start()
    def finish(self):
        self.stop.set();self.thread.join(timeout=10)
        assert not self.thread.is_alive() and not self.errors,self.errors
        return sha(self.directory/'hardware-observations.jsonl')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--power-w',type=int,nargs='+',choices=(150,230,275),default=[230,275])
    args=parser.parse_args();root=args.out.resolve();assert not root.exists()
    assert len(args.power_w)==len(set(args.power_w)), 'Duplicate power caps'
    release=load_release();image=release['image_id']
    assert image=='sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918'
    frozen=REPO/'config/experiments/exl3-migration/full-serving-fixture-manifest.json'
    for name,digest in json.loads(frozen.read_text())['files'].items():assert sha(args.fixture_root/name)==digest,name
    plan=json.loads((REPO/'benchmarks/current-profile-scenarios.json').read_text())
    assert len(plan)==20 and sum(x['repeats'] for x in plan)==70 and sum(x['repeats']*x['concurrency'] for x in plan)==124
    caps=list(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('*/power1_cap'))
    assert len(caps)==1 and int(caps[0].read_text())==180000000
    assert os.access(caps[0],os.W_OK),'User needs write access to power1_cap; no sudo attempted'
    subprocess.run([sys.executable,str(REPO/'scripts/run-server-exl3.py'),'--check-only'],check=True)
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root.mkdir(parents=True)
        frozen_release_sha=sha(REPO/'config/production_image.json')
        state=dict(status='RUNNING',started_unix=time.time(),image_id=image,policy_sha256=release['policy_sha256'],
            original_power_cap_w=180,requested_power_caps_w=args.power_w,variants={},
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            scope='Experimental hardware power override only. Current production policy remains 180 W. No new image or release qualification.')
        save=lambda:write(root/'campaign.json',state)
        save();active=None;monitor=None;success=False
        try:
            subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            running=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
            assert not any(n=='b70-qwen38-vllm' or n.startswith('b70-exl3') for n in running),running
            # Probe requested settings while the card is idle, before any long run.
            for watts in [*args.power_w,180]:set_cap(watts)
            state['power_write_probe']=dict(status='PASS',sudo_required=False,accepted_watts=[*args.power_w,180]);save()
            for watts in args.power_w:
                assert sha(REPO/'config/production_image.json')==frozen_release_sha
                directory=root/f'{watts}w';directory.mkdir();set_cap(watts)
                campaign=dict(status='RUNNING',started_unix=time.time(),image_receipt=release,
                    policy_path='config/production_policy.json',fixture_manifest_sha256=sha(frozen),
                    frozen_fixture_root=str(args.fixture_root.resolve()),power_cap_w=watts,
                    effective_partition_cache_capacity=64,
                    sources={f:sha(REPO/'scripts'/f) for f in ['run-exl3-power-benchmarks.py','exl3_candidate_worker.py','current-profile-benchmark.py','summarize-readme-benchmarks.py']},
                    stages={},scope='Experimental hardware override; image policy still describes qualified 180 W production.')
                state['variants'][str(watts)]=dict(status='RUNNING',path=str(directory.relative_to(REPO)));save()
                monitor=Hardware(directory,watts);monitor.start()
                try:
                    for name,only in [('source-review',None),('prefix-64k-isolated','prefix-64k-cold-warm')]:
                        active=Worker(image,directory/(name+'-worker'),name='b70-exl3-power-worker',env={'EXL3_SDPA_CACHE_CAPACITY':'64'})
                        command=[sys.executable,str(REPO/'scripts/current-profile-benchmark.py'),'--base',BASE,
                            '--container',active.name,'--expected-max-num-seqs','16','--fixture-root',str(args.fixture_root.resolve()),
                            '--legacy-prefix-namespace','--output-root',str(directory/name),'--execute']
                        if only:command+=['--only',only]
                        campaign['stages'][name]=dict(status='RUNNING',command=command);write(directory/'campaign.json',campaign)
                        print('START',watts,name,flush=True)
                        try:
                            active.start();campaign['stages'][name]['loader']=S.loader(active.root);write(directory/'campaign.json',campaign)
                            with (directory/(name+'.log')).open('w') as log:subprocess.run(command,check=True,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                            campaign['stages'][name]['status']='COMPLETE';write(directory/'campaign.json',campaign)
                        finally:active.stop();active=None
                        print('COMPLETE',watts,name,flush=True)
                finally:
                    campaign['hardware_observations_sha256']=monitor.finish();monitor=None
                campaign['status']='COMPLETE_POWER_SERVING_MATRIX';campaign['finished_unix']=time.time();write(directory/'campaign.json',campaign)
                summary=E.summarize(directory,serving_only=True,experimental_power_w=watts)
                summary.update(experimental_power_override_w=watts,qualified_production_power_w=180,
                               power_comparison_scope='Same immutable v2 image and serving profile; only the hardware power cap changes. Not a new qualified production profile.')
                write(directory/'summary.json',summary)
                state['variants'][str(watts)]['status']='COMPLETE';save()
            state['status']='COMPLETE_POWER_COMPARISON_MEASUREMENTS';success=True
        except BaseException as exc:
            state['status']='FAILED';state['error']=repr(exc);raise
        finally:
            state['cleanup_errors']=[]
            for action in ([active.stop] if active is not None else [])+([monitor.finish] if monitor is not None else []):
                try:action()
                except BaseException as exc:state['cleanup_errors'].append(repr(exc))
            try:
                set_cap(180)
                assert sha(REPO/'config/production_image.json')==frozen_release_sha
                state['restoration']=S.restore(image);state['restored_power_cap_w']=180
                if state['cleanup_errors']:state['status']='FAILED_CLEANUP'
            except BaseException as exc:
                state['status']='FAILED_RESTORATION';state['restoration_error']=repr(exc);raise
            finally:
                state['finished_unix']=time.time();state['success']=success and not state['cleanup_errors'];save()
        print(json.dumps(dict(status=state['status'],restored_image=image,restored_power_w=180)),flush=True)


if __name__=='__main__':main()
