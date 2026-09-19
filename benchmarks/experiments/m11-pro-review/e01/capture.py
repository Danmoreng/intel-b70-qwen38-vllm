"""Diagnostic-only real activation capture. Never enabled in timed serving."""
import json
import re
from pathlib import Path
import torch

seen=set()
def capture(module, qkv, positions):
    root=Path('/evidence/activations')
    if not (root/'enabled').exists():return
    prefix=module._b70_capture_prefix
    if not ('mtp' in prefix or re.search(r'\.layers\.(3|35|63)\.',prefix)):return
    m=qkv.shape[0]
    # Bound disk/memory volume; retain complete real split-view strides.
    if m>6656:return
    label=(root/'enabled').read_text().strip()
    key=(prefix,m,label)
    if key in seen:return
    seen.add(key)
    target=root/(re.sub(r'[^a-zA-Z0-9_.-]','_',prefix)+f'-{label}-{m}.pt')
    rope=module.rotary_emb
    payload={'prefix':prefix,'label':label,'qkv':qkv.detach().cpu(),
        'positions':positions.detach().cpu(),'qweight':module.q_norm.weight.detach().cpu(),
        'kweight':module.k_norm.weight.detach().cpu(),
        'eps':module.q_norm.variance_epsilon,'source_qkv_stride':list(qkv.stride()),
        'rope':{'head_size':rope.head_size,'rotary_dim':rope.rotary_dim,
            'max_position_embeddings':rope.cache_max_position_num//4,'base':rope.base,
            'is_neox_style':rope.is_neox_style,'dtype':str(rope.dtype),
            'mrope_section':rope.mrope_section,'mrope_interleaved':rope.mrope_interleaved}}
    torch.save(payload,target)
    cache=root/'cos-sin-cache.pt'
    if not cache.exists():torch.save(rope.cos_sin_cache.detach().cpu(),cache)
    print('B70_E01_REAL_CAPTURE '+json.dumps({'path':str(target),'prefix':prefix,'m':m,'label':label}),flush=True)
