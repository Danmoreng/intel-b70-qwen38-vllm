"""Format guard for GPTQ-specific draft representations. No Torch dependency."""
from dataclasses import dataclass
import os

GPTQ_FORMATS=frozenset(('gptq','auto_gptq','gptq_marlin'))
FLAGS=('B70_MTP_BF16_DRAFT','B70_DRAFT_LMHEAD_INT4','B70_DRAFT_MTP_INT4')

@dataclass(frozen=True)
class MTPPolicy:
    format: str
    is_gptq: bool
    bf16_draft: bool
    lmhead_int4: bool
    mtp_int4: bool
    description: str

def resolve_mtp_policy(quant_config, hf_config=None, environment=None):
    env=os.environ if environment is None else environment
    format=quant_config.get_name() if quant_config is not None else 'unquantized'
    values=[]
    for name in FLAGS:
        value=env.get(name,'0')
        if value not in ('0','1'): raise ValueError(f'{name} must be 0 or 1, got {value!r}')
        values.append(value=='1')
    is_gptq=format in GPTQ_FORMATS
    if any(values) and not is_gptq:
        raise ValueError(f'GPTQ MTP overrides are incompatible with quantization={format}: '+
                         ', '.join(n for n,v in zip(FLAGS,values) if v))
    bf16,head,mtp=values
    hf_qc=getattr(hf_config,'quantization_config',None)
    excluded=isinstance(hf_qc,dict) and any(k.startswith('-:') and 'mtp' in k
                                           for k in hf_qc.get('dynamic',{}))
    if mtp and not (bf16 or excluded):
        raise ValueError('B70_DRAFT_MTP_INT4 requires unquantized source MTP weights (BF16 override or GPTQ dynamic exclusion)')
    description=(f'{format}: GPTQ overrides bf16={int(bf16)}, lmhead_int4={int(head)}, mtp_int4={int(mtp)}'
                 if is_gptq else f'{format}: checkpoint-native MTP; GPTQ overrides disabled')
    return MTPPolicy(format,is_gptq,bf16,head,mtp,description)
