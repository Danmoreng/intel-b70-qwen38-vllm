#!/usr/bin/env python3
"""CPU-only oneDNN source/build/binary mapping; no inference or library swap."""
import argparse,hashlib,json,subprocess
from pathlib import Path
EXPECTED='ee8fc42adcf851ea0f8013ebda24f07629f49ca6ed815835a16da45ad718ca20'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    root=Path('/home/sebastian/LocalLLM');source=root/'b70-onednn-3.13';build=root/'b70-onednn-3.13-build';install=root/'b70-onednn-3.13-install'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)
    assert commit=='0e2a5bfeef1bfbffc3137464606540233086ce9b' and not status
    lib=install/'lib/libdnnl.so.3';assert sha(lib)==EXPECTED
    raw=lib.read_bytes();at=raw.index(b'S_max_slm_size');start=raw.rfind(b'\0',0,at)+1;end=raw.find(b'\0',at)
    embedded=raw[start:end].decode();barrier=embedded.index('intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE)');initialize=embedded.index('S_max_slm[(q')
    assert barrier<initialize,'Unexpected binary kernel ordering; re-audit source ancestry'
    (a.out/'old-embedded-sdpa-micro.cl').write_text(embedded)
    micro=source/'src/gpu/intel/sdpa/micro.cl';(a.out/'old-source-micro.cl').write_bytes(micro.read_bytes())
    (a.out/'old-CMakeCache.txt').write_bytes((build/'CMakeCache.txt').read_bytes())
    options={}
    for line in (build/'CMakeCache.txt').read_text().splitlines():
        if line.startswith(('ONEDNN_','CMAKE_BUILD_TYPE:','CMAKE_CXX_COMPILER:','CMAKE_C_COMPILER:','CMAKE_CXX_FLAGS:','CMAKE_INSTALL_PREFIX:')) and '=' in line:
            k,v=line.split('=',1);options[k]=v
    new=root/'b70-stack-onednn-3.13.4-20261010'
    stable_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=new,text=True).strip()
    new_source=(new/'src/gpu/intel/sdpa/micro.cl').read_text()
    assert new_source.index('S_max_slm[c] = -INFINITY')<new_source.index('intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE)')
    result=dict(status='PROVEN_LOCAL_SOURCE_BUILD_BINARY_MAPPING_OLD_ORDERING',old_commit=commit,old_working_tree=status,
        build_cache_sha256=sha(build/'CMakeCache.txt'),old_library=dict(path=str(lib.resolve()),bytes=lib.stat().st_size,sha256=sha(lib)),
        old_source_sha256=sha(micro),embedded_source_sha256=hashlib.sha256(embedded.encode()).hexdigest(),
        embedded_order=dict(early_q_barrier_offset=barrier,s_max_initialization_offset=initialize),cmake_options=options,
        stable_candidate_commit=stable_commit,stable_micro_sha256=sha(new/'src/gpu/intel/sdpa/micro.cl'),
        binary_scope='The exact local install matches the actual mapped production binary hash; embedded source proves its ordering, not runtime dispatch of every specialization.',
        compile_branch='micro.cpp sets Q_ARRIVE_AWAIT_BARRIER from Q > 1; actual selected micro-SDPA implementation must still be logged.',
        relevant_fix='b38b47c9af53e2ad744241613728fa6cc545ec06')
    (a.out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
