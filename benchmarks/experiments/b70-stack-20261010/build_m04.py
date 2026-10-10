#!/usr/bin/env python3
"""Build only the E04a reduction port against the qualified M04 header pin."""
import argparse,json,subprocess,time
from pathlib import Path
from build_onednn import BUILDER
HERE=Path(__file__).resolve().parent
KERNELS=Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/native-prefill-d-180w/kernel-source')
TLA=Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/optimization/attention-tiles-d-180w/sycl-tla')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    pins={KERNELS:'6d92b1bfbf32767ecda8e819613eb151e70030ad',TLA:'87f6850680a580654b9ea2c80dbc01aeb36ad231'}
    for directory,pin in pins.items():
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,text=True).strip()==pin
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=directory,text=True)
    command=['docker','run','--rm','--name','b70-stack-e04a-build','--network','none','--cpus','4','--memory','16g',
        '-v',str(HERE/'candidates/E04a')+':/src:ro','-v',str(KERNELS)+':/kernels:ro','-v',str(TLA)+':/sycl-tla:ro',
        '-v',str(out)+':/output','--entrypoint','bash',BUILDER,'/src/build-container.sh']
    r=dict(status='RUNNING',command=command,header_pins={str(k):v for k,v in pins.items()},started_unix=time.time())
    def save():(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
    save()
    try:
        with (out/'controller.log').open('w') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT)
        r.update(status='BUILD_CPU_IMPORT_PASS_GPU_UNTESTED',manifest=json.loads((out/'manifest.json').read_text()))
    except BaseException as e:r.update(status='FAILED',error=repr(e));raise
    finally:r['finished_unix']=time.time();save()
    print('E04a_COMPILE_PASS_GPU_UNTESTED',flush=True)
if __name__=='__main__':main()
