#!/usr/bin/env python3
"""Pin a numerically qualified native kernel candidate without weakening guards."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PARENT='sha256:f1822d3b533eb77a3d9f9c3b610bab0c8334c329a40d6ed32041345d79ad567e'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['build','numerics','panel','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent',default=PARENT,choices=[PARENT])
    a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    lib=a.build/'_C.so';manifest=a.build/'_C.so.manifest.json'
    m=json.loads(manifest.read_text());digest=sha(lib)
    assert m['library_sha256']==digest and m['oneDNN_enabled'] and m['torch_version']=='2.14.0+xpu'
    numeric=json.loads(a.numerics.read_text());panel=json.loads(a.panel.read_text())
    assert numeric['status']=='PASS_INDEPENDENT_MATRIX_NUMERICS_GRAPHS_PADDING'
    assert panel['status']=='PASS_ACTUAL_TARGET_MATRIX_PANEL_NOT_AGENT_BENEFIT'
    assert numeric['library_sha256']==panel['library_sha256']==digest
    # Extract the exact parent's plugin; its existing runtime/loader/source guards remain intact.
    name='b70-target-matrix-parent-assets'
    subprocess.run(['docker','create','--name',name,a.parent],stdout=subprocess.DEVNULL,check=True)
    try:subprocess.run(['docker','cp',name+':/opt/exl3xpu/exl3xpu/vllm_plugin.py',str(root/'parent-vllm_plugin.py')],check=True)
    finally:subprocess.run(['docker','rm',name],stdout=subprocess.DEVNULL,check=True)
    parent=root/'parent-vllm_plugin.py';text=parent.read_text()
    anchor='    """vllm.general_plugins entry point (runs in every vLLM process)."""\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'    from . import target_matrix_guard\n    target_matrix_guard.check()\n')
    (root/'vllm_plugin.py').write_text(text)
    shutil.copy2(lib,root/'_C.so');shutil.copy2(manifest,root/'_C.so.manifest.json')
    expected={'_C.so':digest,'_C.so.manifest.json':sha(manifest),'vllm_plugin.py':sha(root/'vllm_plugin.py')}
    (root/'target_matrix_guard.py').write_text('''"""Experimental native-artifact guard, in addition to all original source guards."""
import hashlib
import os
from pathlib import Path
EXPECTED = '''+repr(expected)+'''

def check():
    root=Path(__file__).parent
    override=os.environ.get('EXL3_LIB')
    if override and Path(override).resolve() != (root/'_C.so').resolve():
        raise RuntimeError('Unqualified EXL3_LIB override for target matrix candidate')
    actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in EXPECTED}
    if actual != EXPECTED:raise RuntimeError('Unknown target matrix artifacts: '+str(actual))
''')
    policy=dict(schema=1,status='EXPERIMENT_INDEPENDENT_NUMERICS_PASS_AGENT_BENEFIT_UNQUALIFIED',
        parent_image=a.parent,library_sha256=digest,manifest_sha256=sha(manifest),
        build_flags=m['flags'],source_sha256=m['source_sha256'],
        parent_plugin_sha256=sha(parent),plugin_sha256=sha(root/'vllm_plugin.py'),
        guard_sha256=sha(root/'target_matrix_guard.py'),
        gate_evidence=dict(numerics_sha256=sha(a.numerics),matrix_panel_sha256=sha(a.panel)),
        contract='Only EXL3 native matrix kernels changed. Parent runtime, original guards/sampling/native memory planner/reservations remain. FP8 fixed15 DSpine with selectedKV/replay or original MTP3, according to explicitly selected parent. No automatic production promotion. No use of stale adaptive cost curves.')
    (root/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    digest=sha(root/'policy.json');(root/'policy.sha256').write_text(digest+'  policy.json\n')
    (root/'Dockerfile').write_text('FROM '+a.parent+'\nLABEL org.local.b70.release-status="EXPERIMENT" org.local.b70.policy.sha256="'+digest+'"\n'
        +'COPY _C.so _C.so.manifest.json vllm_plugin.py target_matrix_guard.py /opt/exl3xpu/exl3xpu/\n'
        +'COPY policy.json /opt/b70-dspine/agent-policy.json\n')
    print('Prepared pinned experimental matrix image',digest)


if __name__=='__main__':main()
