#!/usr/bin/env python3
"""Capture the active source-guard closure and V2 integration source, read-only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

CAPTURE = r'''
import ast, hashlib, io, json, sys, tarfile
from pathlib import Path
import vllm
if not __debug__:raise RuntimeError('Runtime guard capture requires ordinary Python execution')
site=Path(vllm.__file__).resolve().parent
workspace=Path('/workspace/vllm/vllm')
plugin=Path('/opt/exl3xpu/exl3xpu')
guards={};definitions={}
for module in ['attention_dispatch.py','target_runtime.py']:
 p=plugin/module;source=p.read_bytes();definitions[module]=source
 tree=ast.parse(source)
 values=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
         and any(isinstance(t,ast.Name) and t.id=='SOURCE_HASHES' for t in n.targets)]
 assert len(values)==1, 'Missing/literal source guard: '+module
 for rel,digest in values[0].items():
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
  assert rel not in guards or guards[rel]['expected_sha256']==digest, 'Conflicting guards: '+rel
  item=guards.setdefault(rel,dict(expected_sha256=digest,declared_by=[]))
  item['declared_by'].append(module)
for rel,item in guards.items():
 p=site/rel;assert p.is_file(), 'Missing declared dependency: '+rel
 actual=hashlib.sha256(p.read_bytes()).hexdigest()
 assert actual==item['expected_sha256'], 'Source guard hash mismatch: '+rel
 item.update(installed_path='installed-vllm/'+rel,actual_sha256=actual)
selected=set(guards)
for folder in ['v1/worker/gpu','v1/core/sched','model_executor/layers/mamba/gdn']:
 selected.update(str(p.relative_to(site)) for p in (site/folder).rglob('*.py'))
for rel in ['v1/core/kv_cache_manager.py','v1/core/kv_cache_utils.py','v1/core/single_type_kv_cache_manager.py',
 'v1/worker/xpu_worker.py','v1/worker/xpu_model_runner_v2.py','compilation/cuda_graph.py',
 'compilation/backends.py','model_executor/models/qwen3_5.py','model_executor/models/qwen3_5_mtp.py',
 'v1/attention/backends/gdn_attn.py','_xpu_ops.py']:
 if (site/rel).is_file():selected.add(rel)
data={};relations=[]
license=Path('/workspace/vllm/LICENSE')
if license.is_file():data['vllm-LICENSE']=license.read_bytes()
for rel in sorted(selected):
 p=site/rel;data['installed-vllm/'+rel]=p.read_bytes()
 wp=workspace/rel
 if wp.is_file():
  other=wp.read_bytes();same=other==data['installed-vllm/'+rel]
  relations.append(dict(path=rel,workspace_sha256=hashlib.sha256(other).hexdigest(),identical=same))
  if not same:data['workspace-vllm/'+rel]=other
 else:relations.append(dict(path=rel,workspace_present=False))
for module,source in definitions.items():data['active-guards/'+module]=source
coverage=dict(status='PASS_ALL_ACTIVE_SOURCE_GUARDS',vllm_version=vllm.__version__,
 required_count=len(guards),covered_count=len(guards),guards=guards,workspace_relationship=relations,
 files_sha256={name:hashlib.sha256(b).hexdigest() for name,b in data.items()})
data['SOURCE_GUARD_COVERAGE.json']=(json.dumps(coverage,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for name,b in sorted(data.items()):
  info=tarfile.TarInfo(name);info.size=len(b);tar.addfile(info,io.BytesIO(b))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container', default='b70-qwen38-vllm')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Fresh capture output directory required')
    identity = json.loads(subprocess.check_output(['docker', 'inspect', a.container]))[0]
    raw = subprocess.check_output(['docker', 'exec', a.container, 'python', '-c', CAPTURE])
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for item in tar:
            name = Path(item.name)
            if not item.isfile() or name.is_absolute() or '..' in name.parts or item.size > 1024*1024:
                raise RuntimeError('Invalid runtime capture entry: ' + item.name)
            data[item.name] = tar.extractfile(item).read()
    coverage = json.loads(data['SOURCE_GUARD_COVERAGE.json'])
    if coverage['required_count'] != coverage['covered_count'] or not coverage['required_count']:
        raise RuntimeError('Incomplete runtime source-guard coverage')
    for name, digest in coverage['files_sha256'].items():
        if hashlib.sha256(data[name]).hexdigest() != digest:
            raise RuntimeError('Capture hash mismatch: ' + name)
    data['CAPTURE_IDENTITY.json'] = (json.dumps(dict(container_id=identity['Id'],
        image_id=identity['Image'],labels=identity['Config']['Labels']),indent=2)+'\n').encode()
    a.output.mkdir(parents=True)
    for name, content in data.items():
        target = a.output/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
    archive = a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, content in sorted(data.items()):
            z.writestr(name, content)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError('Invalid supplement ZIP')
    print(json.dumps(dict(status=coverage['status'],source_guard_files=coverage['required_count'],
                          archive=str(archive),bytes=archive.stat().st_size),indent=2))


if __name__ == '__main__':
    main()
