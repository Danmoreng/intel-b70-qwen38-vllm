"""Regression gates: accept external draft; never hide missing EXL3 target/MTP."""
from types import SimpleNamespace as NS
from torch import nn
from exl3xpu.loader_audit import audit_runner
class Model(nn.Module):
    def __init__(self,names):
        super().__init__()
        for i,name in enumerate(names):
            layer=nn.Module();layer.exl3_loader_report=dict(members=[name],bits=4)
            self.add_module('linear_'+str(i),layer)
class Config:
    expected_inventory=[dict(canonical=k,bits=4) for k in ['target','lm_head','mtp.ffn']]
    def get_name(self):return 'exl3'
def runner(method,target,draft):
    return NS(vllm_config=NS(quant_config=Config(),speculative_config=NS(method=method) if method else None),model=Model(target),speculator=NS(model=Model(draft)) if draft is not None else None)
assert audit_runner(runner('dflash',['target','lm_head'],['lm_head']))['mtp_expected']==0
assert audit_runner(runner('mtp',['target','lm_head'],['mtp.ffn','lm_head']))['mtp_expected']==1
assert audit_runner(runner(None,['target','lm_head'],None))['expected_modules']==2
for name,args in [('missing-target',('dflash',['lm_head'],['lm_head'])),('missing-mtp',('mtp',['target','lm_head'],['lm_head'])),('mtp-not-instantiated',('mtp',['target','lm_head'],None)),('duplicate-target',('dflash',['target','target','lm_head'],['lm_head'])),('unexpected-tensor',('dflash',['target','lm_head','unknown'],['lm_head']))]:
    try:audit_runner(runner(*args))
    except ValueError:print('PASS_REJECTION',name)
    else:raise AssertionError(name)
print('PASS_EXTERNAL_DRAFT_INVENTORY_GATES')
