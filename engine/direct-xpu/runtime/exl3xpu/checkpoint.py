"""Authoritative, dependency-free EXL3 inventory shared by loader and gates."""
from __future__ import annotations
import json
from pathlib import Path
import re

GPTQ_DRAFT_FLAGS=('B70_MTP_BF16_DRAFT','B70_DRAFT_LMHEAD_INT4','B70_DRAFT_MTP_INT4')

def validate_exl3_environment(environment):
    for name in GPTQ_DRAFT_FLAGS:
        value=environment.get(name,'0')
        if value not in ('0','1'): raise ValueError(f'{name} must be 0 or 1')
        if value=='1': raise ValueError(f'{name} is a GPTQ representation override and cannot be used with EXL3')

def normalize_key(key: str) -> str:
    parts=key.split('.')
    if 'mtp' in parts: return 'mtp.'+'.'.join(parts[parts.index('mtp')+1:])
    match=re.search(r'(layers\.\d+\..*)$',key)
    return match.group(1) if match else parts[-1]

def inventory(weight_map: dict, quantization: dict, *, bits=4, head_bits=6,
              mtp_bits=4, codebook='mul1') -> list[dict]:
    stored=quantization.get('tensor_storage',{})
    modules=[]; seen=set()
    for name in sorted(weight_map):
        if not name.endswith('.trellis'): continue
        key=name[:-len('.trellis')]; canonical=normalize_key(key)
        if canonical in seen: raise ValueError(f'Duplicate canonical EXL3 module: {canonical}')
        seen.add(canonical)
        meta=stored.get(key,{})
        if meta and meta.get('quant_format')!='exl3':
            raise ValueError(f'Index/quantization format conflict: {key}')
        default=head_bits if canonical=='lm_head' else mtp_bits if canonical.startswith('mtp.') else bits
        width=int(meta.get('bits_per_weight',default))
        cb='mcg' if key+'.mcg' in weight_map else 'mul1' if key+'.mul1' in weight_map else '3inst'
        if cb!=codebook: raise ValueError(f'Codebook mismatch for {key}: {cb} != {codebook}')
        required=[key+s for s in ('.trellis','.suh','.svh')]
        if cb!='3inst': required.append(key+'.'+cb)
        missing=[n for n in required if n not in weight_map]
        if missing: raise ValueError(f'Incomplete EXL3 module {key}: {missing}')
        if width<=0: raise ValueError(f'Invalid bit width for {key}: {width}')
        modules.append({'key':key,'canonical':canonical,'bits':width,'codebook':cb,
                        'mtp':canonical.startswith('mtp.'),'tensors':required,
                        'bit_width_source':'tensor_storage' if meta else 'checkpoint defaults'})
    omitted=[k for k,v in stored.items() if v.get('quant_format')=='exl3' and k+'.trellis' not in weight_map]
    if omitted: raise ValueError(f'Quantized modules missing from weight index: {omitted}')
    if not modules: raise ValueError('No EXL3 trellis tensors in authoritative weight index')
    return modules

def load_inventory(model) -> tuple[dict,list[dict]]:
    root=Path(model)
    wmap=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    qc=json.loads((root/'quantization_config.json').read_text())
    return wmap,inventory(wmap,qc,bits=int(qc.get('bits',4)),head_bits=qc.get('head_bits',6),
                          mtp_bits=qc.get('mtp_bits',int(qc.get('bits',4))),codebook=qc.get('codebook','mul1'))
