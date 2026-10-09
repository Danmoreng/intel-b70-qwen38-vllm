"""Replay actual captured KV specs through the installed grouping function."""
import dataclasses
import json
import os
from pathlib import Path
import torch
import vllm.v1.kv_cache_interface as I
from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_uniform_page_size as group

ledger=json.loads(Path('/ledger.json').read_text())
specs={}
for row in ledger['contexts']['96000']['groups']:
    for name,entry in row['layers'].items():
        record=dict(entry['spec']); cls=getattr(I,record.pop('class'))
        if 'dtype' in record:record['dtype']=getattr(torch,record['dtype'].removeprefix('torch.'))
        if 'dtypes' in record:record['dtypes']=tuple(getattr(torch,d.removeprefix('torch.')) for d in record['dtypes'])
        if 'shapes' in record:record['shapes']=tuple(tuple(s) for s in record['shapes'])
        fields={f.name for f in dataclasses.fields(cls)}
        assert set(record)<=fields,(name,set(record)-fields)
        specs[name]=cls(**record)
assert len(specs)==69
os.environ.pop('B70_DFLASH_GROUP4',None)
original=group(specs);assert len(original)==15
os.environ['B70_DFLASH_GROUP4']='1'
candidate=group(specs);assert len(candidate)==18
counts={}
seen=[]
for g in candidate:
    counts[type(g.kv_cache_spec).__name__]=counts.get(type(g.kv_cache_spec).__name__,0)+1
    assert len(g.layer_names)<=4
    for name in g.layer_names:
        assert specs[name]==g.kv_cache_spec
        seen.append(name)
assert counts==dict(MambaSpec=12,FullAttentionSpec=4,SlidingWindowSpec=2),counts
assert len(seen)==len(set(seen))==69 and set(seen)==set(specs)
bad=dict(specs);bad.pop(next(name for name in specs if 'layers.64.' in name))
try:group(bad)
except AssertionError:pass
else:raise AssertionError('Unexpected geometry silently accepted')
result=dict(status='PASS_GROUP4_ACTUAL_SPEC_CPU_REPLAY',counts=counts,layers=69,
    padding_layers=3,all_specs_preserved=True,negative_geometry_rejected=True,
    scope='Grouping only; no GPU capacity, scheduler or prefix-state qualification')
print(json.dumps(result,indent=2))
