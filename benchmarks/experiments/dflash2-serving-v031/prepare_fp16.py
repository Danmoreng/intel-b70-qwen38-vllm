#!/usr/bin/env python3
"""Convert pinned BF16 storage once on CPU, without any quantization."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
SOURCE_SHA='67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=Path('/source');destination=Path('/destination')
assert sha(source/'model.safetensors')==SOURCE_SHA
assert not (destination/'model.safetensors').exists(), 'refuse to overwrite prepared draft'
weights={}
with safe_open(source/'model.safetensors',framework='pt',device='cpu') as f:
    for name in f.keys():
        weights[name]=f.get_tensor(name).to(torch.float16)
        assert torch.isfinite(weights[name]).all(), name
count=sum(t.numel() for t in weights.values())
assert count==1924404480
save_file(weights,destination/'model.safetensors',metadata={'format':'pt','source_sha256':SOURCE_SHA,'conversion':'BF16 to FP16 on CPU; no quantization'})
(destination/'model.safetensors').chmod(0o644)
config=json.loads((source/'config.json').read_text());config['dtype']='float16';config.pop('torch_dtype',None)
(destination/'config.json').write_text(json.dumps(config,indent=2)+'\n')
receipt=dict(kind='unquantized FP16',source_repository='incoai/Qwen3.8-27B-DFlash2',source_revision='dedf8df68adfb1afeaf7b7480c0a0243108177b4',source_sha256=SOURCE_SHA,parameters=count,weight_gib=count*2/2**30,checkpoint_bytes=(destination/'model.safetensors').stat().st_size,fp16_sha256=sha(destination/'model.safetensors'),config_sha256=sha(destination/'config.json'),runtime_bf16_tensors=False)
(destination/'conversion.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
