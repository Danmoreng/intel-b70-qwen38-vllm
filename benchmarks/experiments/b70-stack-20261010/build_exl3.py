#!/usr/bin/env python3
"""E01 native rebuild against matching oneDNN, then isolated qualified-image child."""
import argparse, hashlib, json, shutil, subprocess, time
from pathlib import Path
from build_onednn import BUILDER

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/sebastian/LocalLLM/exl3xpu-vllm031')
COMMIT = 'fc8f8c9fa1627f28a7fbe6ee4fc7bf961dabef81'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--onednn', type=Path, required=True); p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    base = json.loads((HERE/'baseline-manifest.json').read_text())['production']['image_id']
    dnnl = a.onednn.resolve(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip() == COMMIT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
    library = json.loads((dnnl/'library-receipt.json').read_text())
    assert library['version'] == '3.13.4' and sha(dnnl/'install/lib/libdnnl.so.3') == library['library_sha256']
    # Preserve build-time and runtime RPATH /opt/b70/lib; production also uses
    # /opt/b70/libdnnl.so.3. Both copies below have identical immutable bytes.
    command = ['docker','run','--rm','--network','none','--cpus','4','--memory','16g',
        '-v',str(SOURCE)+':/opt/exl3xpu:ro','-v',str(dnnl/'install')+':/opt/b70:ro','-v',str(out)+':/results',
        '-e','EXL3_DNNL_ROOT=/opt/b70','-e','EXL3_OUT=/results/_C.so','-e','EXL3_BUILD_LOG=/results/build.log',
        '--entrypoint','bash',BUILDER,'-c',
        'test -f /opt/b70/include/oneapi/dnnl/dnnl_sycl.hpp && bash /opt/exl3xpu/scripts/build_ext.sh']
    receipt = dict(status='RUNNING',base_image=base,builder=BUILDER,source_commit=COMMIT,command=command,started_unix=time.time())
    def save(): (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        with (out/'controller.log').open('w') as log: subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        native = json.loads((out/'_C.so.manifest.json').read_text())
        assert native['oneDNN_enabled'] and native['torch_version']=='2.14.0+xpu'
        assert set(native['oneDNN_library_sha256'].values()) == {library['library_sha256']}
        stage=out/'image';stage.mkdir()
        for name in ['_C.so','_C.so.manifest.json']:shutil.copy2(out/name,stage/name)
        shutil.copy2(dnnl/'install/lib/libdnnl.so.3',stage/'libdnnl.so.3')
        shutil.copytree(dnnl/'install/include',stage/'include')
        (stage/'Dockerfile').write_text(f'''FROM {base}
LABEL org.local.b70.experiment="E01-oneDNN-3.13.4" org.local.b70.release-status="EXPERIMENTAL_UNQUALIFIED"
COPY _C.so _C.so.manifest.json /opt/exl3xpu/exl3xpu/
COPY libdnnl.so.3 /opt/b70/libdnnl.so.3
COPY libdnnl.so.3 /opt/b70/lib/libdnnl.so.3
COPY include /opt/b70/include/
RUN python /opt/exl3xpu/scripts/verify_native_artifact.py
''')
        tag='local/b70-mtp3-stack:e01-onednn3134-20261010'
        with (out/'image-build.log').open('w') as log:subprocess.run(['docker','build','--network','none','--pull=false','-t',tag,str(stage)],check=True,stdout=log,stderr=subprocess.STDOUT)
        image=subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
        receipt.update(status='BUILD_IMPORT_PASS_GPU_UNTESTED',native=native,image_tag=tag,image_id=image,oneDNN=library)
    except BaseException as e:receipt.update(status='FAILED',error=repr(e));raise
    finally:receipt['finished_unix']=time.time();save()
    print('E01_IMAGE_CPU_IMPORT_PASS_GPU_UNTESTED',image,flush=True)
if __name__=='__main__':main()
