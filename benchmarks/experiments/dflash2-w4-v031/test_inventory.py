"""CPU negative gates against the actual converted draft inventory."""
import copy
import json
from pathlib import Path
from exl3xpu.draft_w4 import DraftW4Config,local_key
from vllm.model_executor.layers.linear import LinearBase,UnquantizedLinearMethod

cfg=json.loads(Path('/draft/config.json').read_text())
inventory=cfg['quantization_config']['module_inventory']
q=DraftW4Config(inventory)
assert local_key('model.layers.64.self_attn.qkv_proj')=='layers.0.self_attn.qkv_proj'
assert local_key('model.layers.68.mlp.down_proj')=='layers.4.mlp.down_proj'
assert local_key('model.fc')=='fc'
for prefix in ['language_model.model.layers.64.self_attn.qkv_proj',
               'model.layers.0.self_attn.q_proj','model.layers.69.mlp.down_proj',
               'fc','model.surprise']:
    try:local_key(prefix)
    except ValueError:pass
    else:raise AssertionError(prefix)
for key,value in [('bits',5),('group_size',64),('role','target')]:
    bad=copy.deepcopy(inventory);bad['fc'][key]=value
    try:DraftW4Config(bad)
    except AssertionError:pass
    else:raise AssertionError((key,value))
bad=dict(inventory);bad.pop('fc')
try:DraftW4Config(bad)
except AssertionError:pass
else:raise AssertionError('Missing module accepted')
layer=object.__new__(LinearBase)
for i in range(64,69):
    for conv in ['attention_conv','mlp_conv']:
        assert isinstance(q.get_quant_method(layer,f'model.layers.{i}.{conv}.kernel_projection'),UnquantizedLinearMethod)
try:q.get_quant_method(layer,'model.layers.64.attention_conv.surprise_projection')
except ValueError:pass
else:raise AssertionError('Unknown module silently dense')
assert len(q.get_quant_method(layer,'model.layers.64.self_attn.qkv_proj').members)==3
assert len(q.get_quant_method(layer,'model.layers.68.mlp.gate_up_proj').members)==2
print(json.dumps(dict(status='PASS_STRICT_DRAFT_ROLE_AND_NAMESPACE_NEGATIVE_GATES',
    source_revision=cfg.get('_name_or_path'),modules=36),indent=2))
