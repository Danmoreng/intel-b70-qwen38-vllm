#!/usr/bin/env python3
"""Small repeated ABBA check; immutable runtime, fixed MTP3, no kernel sweep."""
import argparse
import fcntl
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import time

from exl3_candidate_worker import BASE, REPO, Worker, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--quality-gate',type=Path,required=True)
    p.add_argument('--operations-gate',type=Path,required=True)
    a=p.parse_args();root=a.out.resolve()
    assert not root.exists()
    spec=importlib.util.spec_from_file_location('final_perf_mtp',REPO/'scripts/run-exl3-mtp-study.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    image=subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        q=json.loads((a.quality_gate/'campaign.json').read_text())
        o=json.loads((a.operations_gate/'campaign.json').read_text())
        assert q['status']=='COMPLETE_MEASURED_FINAL_QUALITY_REQUIRES_REVIEW' and q['image_id']==image
        assert o['status']=='COMPLETE_FINAL_OPERATIONAL_GATES' and o['image_id']==image
        raw=gzip.decompress(a.panel.read_bytes());panel=json.loads(raw)
        root.mkdir();m.R.cap()
        state=dict(status='RUNNING',image_id=image,started_unix=time.time(),order=[False,True,True,False],
                   panel_gzip_sha256=sha(a.panel),source_sha256=sha(Path(__file__)),arms=[],
                   scope='Repeated C1 103K cold/warm native versus complete optimized attention. Fixed MTP3; final full matrix separately covers C4.')
        def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
        save()
        try:
            for i,enabled in enumerate(state['order']):
                phase=root/f'arm-{i}-optimized{int(enabled)}'
                w=Worker(image,phase,env={'EXL3_GUARDED_ATTN':str(int(enabled))})
                arm=dict(optimized=enabled,waves=[]);state['arms'].append(arm);save()
                try:
                    w.start()
                    short=next(x for x in panel['windows'] if x['name']=='code-4096')
                    long=next(x for x in panel['windows'] if x['name']=='code-102752')
                    m.wave(short,1,'warmup',i,3,phase,64)
                    for mode in ['cold','warm']:
                        print('START',i,'optimized',enabled,mode,flush=True)
                        result=m.wave(long,1,mode,i,3,phase,512)
                        assert result['native']['preemptions']==0,result['native']
                        arm['waves'].append(result);save()
                        print('DONE',result['native_request_weighted_decode_tps'],flush=True)
                    arm['status']='COMPLETE';save()
                finally:w.stop()
            state['status']='COMPLETE_REPEATED_LONG_PERFORMANCE'
        except BaseException as exc:
            state['status']='FAILED';state['error']=repr(exc);raise
        finally:state['finished_unix']=time.time();save()


if __name__=='__main__':main()
