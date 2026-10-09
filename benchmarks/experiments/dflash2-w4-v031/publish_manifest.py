"""Hash the final public evidence and current recipes after the decision."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'benchmarks/results/dflash2-pro-optimization-v031'


def normalize_diagnostic_json():
    """Publish valid JSON without altering immutable raw diagnostic records."""
    receipt=OUT/'json-normalization.json'
    rows=json.loads(receipt.read_text())['files'] if receipt.exists() else []
    for path in sorted(OUT.rglob('*.json')):
        if path.name in ['PUBLICATION_MANIFEST.json','json-normalization.json']:continue
        changes=[]
        def visit(value,location='$'):
            if isinstance(value,float) and not math.isfinite(value):
                token='NaN' if math.isnan(value) else ('Infinity' if value>0 else '-Infinity')
                changes.append(dict(json_path=location,representation=token))
                return token
            if isinstance(value,dict):return {k:visit(v,location+'.'+k) for k,v in value.items()}
            if isinstance(value,list):return [visit(v,location+f'[{i}]') for i,v in enumerate(value)]
            return value
        original=path.read_bytes();value=visit(json.loads(original))
        if changes:
            encoded=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
            path.write_bytes(encoded)
            rows.append(dict(path=str(path.relative_to(REPO)),
                before_sha256=hashlib.sha256(original).hexdigest(),
                after_sha256=hashlib.sha256(encoded).hexdigest(),fields=changes))
    if rows:
        receipt.write_text(json.dumps(dict(
            scope='Public JSON serialization only; immutable local raw files and measured decisions are unchanged.',
            representation='Non-finite diagnostic scalars are the strings Infinity, -Infinity or NaN, never silently null or omitted.',
            observed_meaning='Negative-infinite acceptance log margins represent zero target mass; infinite winner gaps can represent a single eligible token. These diagnostic values are not evidence of serving activation overflow.',
            files=rows),indent=2,allow_nan=False)+'\n')


def main():
    decision=json.loads((OUT/'decision.json').read_text())
    assert decision['scope']=='BEST_WITHIN_TESTED_CANDIDATES_NOT_GLOBAL_OPTIMUM'
    assert decision['production_restoration']['models_endpoint_healthy']
    old=subprocess.check_output(['git','show','d857bd1:README.md'],cwd=REPO,text=True)
    current=(REPO/'README.md').read_text()
    def power(text):return text.split('<!-- BEGIN POWER COMPARISON -->',1)[1].split('<!-- END POWER COMPARISON -->',1)[0]
    assert power(old)==power(current),'Historical power section was unexpectedly changed'
    normalize_diagnostic_json()
    paths=[REPO/'README.md']
    for name in ['dflash2-w4-v031','mtp4-pro-qualification-v031','dflash2-pro-diagnostics-v031','dflash2-serving-v031']:
        paths.extend(p for p in (REPO/'benchmarks/experiments'/name).iterdir() if p.is_file())
    paths.extend(p for p in OUT.rglob('*') if p.is_file() and p.name!='PUBLICATION_MANIFEST.json')
    records=[dict(path=str(p.relative_to(REPO)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(set(paths))]
    result=dict(schema=1,scope=decision['scope'],files=records,
        historical_power_section_sha256=hashlib.sha256(power(current).encode()).hexdigest(),
        power_remeasured=False,
        raw_data_scope='Raw tensor captures, full prompts, output events and immutable executed snapshots stay local under benchmark-results/. Public raw manifests preserve their paths and SHA256 values.',
        source_scope='This manifest hashes current recipes. Executed source snapshots and image-resident module hashes are separately recorded per campaign; newer source recipes do not inherit old image qualification.')
    (OUT/'PUBLICATION_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PUBLISHED_FINAL_EVIDENCE_MANIFEST',len(records),'HISTORICAL_POWER_UNCHANGED')


if __name__=='__main__':main()
