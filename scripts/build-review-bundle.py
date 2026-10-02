#!/usr/bin/env python3
"""Compact review snapshot with mandatory active runtime-source closure.

Contains current tracked code, explicitly named new files and compact review
receipts. Never includes weights, compiled libraries or raw logit arrays.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import zipfile

REPO=Path(__file__).resolve().parents[1]
TEXT={'.py','.cpp','.c','.cc','.cxx','.cu','.rs','.h','.hpp','.sycl','.sh','.json','.jsonl','.md','.yaml','.yml','.toml','.txt',
      '.patch','.example','.sha256','.in','.pth','.rules','.cmake','.lock','.js','.mjs','.ts','.css','.html'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exl3-source',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime-sources',type=Path,help='Reuse a hash-verified capture from the same immutable qualified image')
    a=p.parse_args();assert not a.output.exists()
    capture=a.runtime_sources or a.output.parent/(a.output.stem+'-runtime')
    # Collector derives dependencies from the actual installed guards and fails
    # before producing a bundle if any guarded source is absent or mismatched.
    if a.runtime_sources:
        identity=json.loads((capture/'CAPTURE_IDENTITY.json').read_text())
        assert identity['image_id']==json.loads((REPO/'config/production_image.json').read_text())['image_id']
    else:
        subprocess.run([sys.executable,str(REPO/'scripts/collect-review-runtime.py'),'--output',str(capture)],check=True)
    data={};origins={};omissions=[]
    def add(name,value,origin):
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        value.decode('utf-8');assert b'\0' not in value
        if re.search(rb'\b(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{30,}|github_pat_[A-Za-z0-9_]{35,})\b',value):
            raise RuntimeError('Potential credential in '+name)
        assert len(value)<=1024*1024,(name,len(value))
        assert name not in data
        data[name]=value;origins[name]=origin
    repos={}
    for role,root in [('engine-code',REPO),('exl3-candidate',a.exl3_source.resolve())]:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        status=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
        repos[role]=dict(commit=head,working_tree_status=status,scope='Working source snapshot; production identity is separate')
        names=subprocess.check_output(['git','ls-files','-co','--exclude-standard','-z'],cwd=root).decode().split('\0')
        for name in sorted(set(filter(None,names))):
            path=root/name
            allowed=(path.suffix in TEXT or path.name.startswith('Dockerfile.') or path.name in {'LICENSE','NOTICE','Dockerfile','CMakeLists.txt','.gitignore','.dockerignore'})
            historical=(role=='engine-code' and ((name.startswith('benchmarks/results/') and not name.startswith(('benchmarks/results/exl3-migration/','benchmarks/results/exl3-review-20261002/','benchmarks/results/exl3-review-release-v2/')))
                or (name.startswith('benchmarks/runs/') and not name.startswith(('benchmarks/runs/2026-10-02-exl3-production/','benchmarks/runs/2026-10-02-flappybird-v7/')) and path.name!='summary.json')
                or (name.startswith('benchmarks/web-coding-fixture/v') and not name.startswith('benchmarks/web-coding-fixture/v7/'))))
            redundant=(path.name in {'runtime-environment.json','trace-summary.json','baseline_environment.json'} or path.name.startswith('loader-') or name in {
                'benchmarks/results/exl3-migration/target-quality-full-v1/comparison.json',
                'benchmarks/results/exl3-migration/target-quality-stages-v1/comparison.json'})
            if not allowed or path.stat().st_size>1024*1024 or historical or redundant or name=='benchmarks/meaningful-corpus.json' or name.startswith(('fixtures/','benchmark-results/')) or '/trace/' in name or '.sse.' in name:
                omissions.append(dict(path=role+'/'+name,bytes=path.stat().st_size,reason='Binary, raw/large fixture or trace outside compact source review'))
                continue
            add(role+'/'+name,path.read_bytes(),str(root/name))
        patch=subprocess.check_output(['git','diff','--binary',head,'--','.',':!engine/exl3xpu/source.tar.gz'],cwd=root)
        add('changes/'+role+'.patch',patch,'Working tracked diff against '+head+'; new files included directly')
    qualified=json.loads((REPO/'engine/exl3xpu/manifest.json').read_text())['source_commit']
    archive_bytes=subprocess.check_output(['git','archive','--format=tar',qualified],cwd=a.exl3_source)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        for item in archive:
            if not item.isfile():continue
            path=Path(item.name)
            if (path.suffix in TEXT or path.name.startswith('Dockerfile.') or path.name in {'LICENSE','NOTICE','Dockerfile','CMakeLists.txt','.gitignore','.dockerignore'}) and item.size<=1024*1024:
                add('exl3-qualified-source/'+item.name,archive.extractfile(item).read(),'Qualified EXL3 git commit '+qualified)
    repos['exl3-qualified-source']=dict(commit=qualified,scope='Current qualified source from the published manifest; archived experiments retain their original identities')
    for path in sorted(capture.rglob('*')):
        if path.is_file():add('qualified-runtime/'+str(path.relative_to(capture)),path.read_bytes(),'Read-only active runtime capture')
    for path in sorted(a.evidence.rglob('*')):
        if not path.is_file():continue
        redundant=(path.name.endswith('-before.json') or path.name=='progress.json')
        failed_partial=(any(part in {'matched-state-v1','matched-state-v2','matched-state-v3','matched-state-v4'} for part in path.parts) and path.name!='campaign.json')
        historical_coverage=(path.relative_to(a.evidence)==Path('runtime-sources/SOURCE_GUARD_COVERAGE.json'))
        unsupported=((path.suffix not in {'.json','.md'} and not ('diagnostic-sources' in path.parts and path.suffix=='.py')) or path.stat().st_size>1024*1024 or ('runtime-sources' in path.parts and not historical_coverage) or '/loader/' in str(path))
        if redundant or failed_partial or unsupported:
            reason=('Superseded progress snapshot' if redundant else
                    'Failed setup partial; campaign outcome retained' if failed_partial else
                    'Raw/binary/large diagnostic or duplicate runtime source; compact receipts retained')
            omissions.append(dict(path='review-evidence/'+str(path.relative_to(a.evidence)),bytes=path.stat().st_size,reason=reason))
            continue
        add('review-evidence/'+str(path.relative_to(a.evidence)),path.read_bytes(),'Local diagnostic receipt; instrumented tests not serving benchmark')
    def meta(name,value):add(name,(json.dumps(value,indent=2)+'\n').encode(),'Bundle metadata')
    coverage=json.loads(data['qualified-runtime/SOURCE_GUARD_COVERAGE.json'])
    assert coverage['status']=='PASS_ALL_ACTIVE_SOURCE_GUARDS' and coverage['required_count']==coverage['covered_count']
    for name,digest in coverage['files_sha256'].items():
        assert hashlib.sha256(data['qualified-runtime/'+name]).hexdigest()==digest
    for item in coverage['guards'].values():
        name='qualified-runtime/'+item['installed_path']
        assert hashlib.sha256(data[name]).hexdigest()==item['expected_sha256']
    build=json.loads((REPO/'engine/exl3xpu/review-candidate/build-receipt.json').read_text())
    for name,digest in build['source_sha256'].items():
        assert hashlib.sha256(data['exl3-candidate/'+name]).hexdigest()==digest, 'Candidate build input missing or changed: '+name
    old_coverage=data.get('review-evidence/runtime-sources/SOURCE_GUARD_COVERAGE.json')
    if old_coverage is not None:
        assert hashlib.sha256(old_coverage).hexdigest()=='51ad7a2b21a6d6c87fc08361ce83d26d5b5937018c3fa07eaa028aafd489b944'
    meta('SOURCE_PROVENANCE.json',dict(repositories=repos,source_guard_count=coverage['required_count'],
        qualified_runtime=json.loads(data['qualified-runtime/CAPTURE_IDENTITY.json']),
        scope='Current owned code and EXL3 candidate code, complete active source-guard closure, compact diagnostic receipts. No weights, binaries, raw logits or new production claim.'))
    meta('OMISSIONS.json',omissions)
    meta('MANIFEST.json',dict(files=[dict(path=name,bytes=len(value),sha256=hashlib.sha256(value).hexdigest(),origin=origins[name]) for name,value in sorted(data.items())]))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    encoded=io.BytesIO()
    with zipfile.ZipFile(encoded,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,value in sorted(data.items()):archive.writestr(name,value)
    with zipfile.ZipFile(encoded) as archive:
        sizes=sorted([dict(path=i.filename,compressed_bytes=i.compress_size,bytes=i.file_size) for i in archive.infolist()],key=lambda x:x['compressed_bytes'],reverse=True)
    if encoded.tell()>5*1024*1024:
        print(json.dumps(dict(archive_bytes=encoded.tell(),largest_compressed_entries=sizes[:15]),indent=2))
        raise RuntimeError('Review archive exceeds 5 MiB; do not silently omit required source')
    with zipfile.ZipFile(encoded) as archive:
        assert archive.testzip() is None
        for name,value in data.items():assert archive.read(name)==value
    a.output.write_bytes(encoded.getvalue())
    digest=hashlib.sha256(a.output.read_bytes()).hexdigest()
    a.output.with_suffix(a.output.suffix+'.sha256').write_text(digest+'  '+a.output.name+'\n')
    print(json.dumps(dict(archive=str(a.output),bytes=a.output.stat().st_size,files=len(data),sha256=digest,active_guard_files=coverage['required_count']),indent=2))


if __name__=='__main__':main()
