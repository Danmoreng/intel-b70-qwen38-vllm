#!/usr/bin/env python3
"""Final immutable profile capacity/extension/abort/media and worker-restart gates."""
import argparse
import base64
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import time
import zlib

from exl3_candidate_worker import Worker, REPO, BASE, sha

spec=importlib.util.spec_from_file_location('final_operations_helpers',REPO/'scripts/run-exl3-contract-diagnostics.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


def unique_red(index):
    """32 distinct capped-area images; no multimodal cache deduplication shortcut."""
    width=height=2048
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
    scan=b'\0'+bytes([255,index,0])*width
    data+=chunk(b'IDAT',zlib.compress(scan*height))+chunk(b'IEND',b'')
    return dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(data).decode()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mixed-gate',type=Path,required=True);p.add_argument('--rows-gate',type=Path,required=True)
    args=p.parse_args();root=args.out.resolve();assert not root.exists()
    with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);D.R.cap()
        mixed=json.loads((args.mixed_gate/'campaign.json').read_text())
        rows=json.loads((args.rows_gate/'probe/result.json').read_text())
        image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
        assert mixed['status']=='COMPLETE_MIXED_AND_API_MEDIA_GATES' and mixed['image_id']==image
        assert rows['status']=='PASS_EXPANDED_ROW_PADDING_GRAPH_GATE'
        root.mkdir();state=dict(status='RUNNING',started_unix=time.time(),image_id=image,cases={},
            source_sha256={f:sha(REPO/'scripts'/f) for f in ['run-exl3-final-operations.py','exl3_candidate_worker.py','run-exl3-contract-diagnostics.py']},
            mixed_gate_sha256=sha(args.mixed_gate/'campaign.json'),rows_gate_sha256=sha(args.rows_gate/'probe/result.json'),
            c16_preemptions_allowed=True)
        def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
        save();subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True)
        worker=Worker(image,root/'serving',env={'EXL3_MIGRATION_TRACE':'/results/trace'})
        try:
            worker.start()
            for case in ['139k','188k','near-limit','c4-long','c16-long','image-long','extension-abort-long']:
                path=root/case;path.mkdir();print('START',case,flush=True)
                result=(D.long_case(case,path,262144) if case in ('139k','188k','near-limit') else D.operational(case,path,True))
                assert result['status'] in ('PASS','COMPLETE')
                if case in ('139k','188k','near-limit'):assert result['native']['preemptions']==0
                state['cases'][case]=result;save();print('PASS',case,flush=True)
            print('START independent maximum-area images',flush=True)
            before=D.R.snapshot(BASE);start=time.monotonic()
            pictures=[unique_red(i) for i in range(32)]
            response=D.R.http(BASE,'/v1/chat/completions',D.chat(pictures+[
                dict(type='text',text='What main color is shared by all32 pictures? Reply with one word.')]),timeout=1800)
            _,native=D.R.wait_accounted(BASE,before,response['usage'])
            assert 'red' in response['choices'][0]['message']['content'].lower()
            assert native['preemptions']==0 and response['usage']['prompt_tokens']>100000
            state['cases']['independent32-max-area-images']=dict(status='PASS',usage=response['usage'],native=native,
                wall_s=time.monotonic()-start,unique_images=32,pixels_each=4194304,
                fixture_sha256=[hashlib.sha256(x['image_url']['url'].encode()).hexdigest() for x in pictures])
            save()
        except BaseException as exc:state['status']='FAILED';state['error']=repr(exc);raise
        finally:worker.stop();save()
        try:
            # Require the loader's full inventory on the final packaged image.
            reports=list((root/'serving/loader').glob('loader-*.json'));assert len(reports)==1
            state['loader_report_sha256']=sha(reports[0])
            loader=json.loads(reports[0].read_text())
            assert (loader['status']=='PASS' and loader['loaded_modules']==loader['expected_modules']==409
                    and loader['mtp_loaded']==loader['mtp_expected']==8 and not loader['missing']
                    and not loader['unexpected_duplicates']), 'Incomplete final loader inventory'
            state['loader']=loader
            trace_files=list((root/'serving/trace').glob('*'))
            aborted=0
            for path in trace_files:
                if path.is_file() and path.suffix in ('.jsonl','.json'):
                    aborted+=path.read_text().count('FINISHED_ABORTED')
            assert aborted>0,'Scheduler did not confirm abort'
            state['confirmed_abort_trace_occurrences']=aborted
            restart=Worker(image,root/'restarted')
            try:
                restart.start();response=D.R.http(BASE,'/v1/chat/completions',D.chat('Reply with the capital of France, one word.'),timeout=180)
                assert 'paris' in response['choices'][0]['message']['content'].lower()
                state['cases']['worker-restart']=dict(status='PASS',reply=response['choices'][0]['message']['content'])
            finally:restart.stop()
            state['status']='COMPLETE_FINAL_OPERATIONAL_GATES'
        except BaseException as exc:state['status']='FAILED';state['error']=repr(exc);raise
        finally:
            state['production_left_offline']=True;state['finished_unix']=time.time();save()


if __name__=='__main__':main()
