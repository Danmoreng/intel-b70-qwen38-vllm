"""Explicit draft-only W4/G128 inventory and oneDNN XPU packed linear method."""
import hashlib
import json
import os
import re
import weakref

import torch
from torch.nn import Parameter
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

FUSED = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'], 'gate_up_proj': ['gate_proj', 'up_proj']}
QKV_IDS = {'q': 0, 'k': 1, 'v': 2}


def local_key(prefix):
    match = re.fullmatch(r'model\.layers\.(6[4-8])\.(self_attn|mlp)\.(\w+)', prefix)
    if match:
        return f'layers.{int(match[1])-64}.{match[2]}.{match[3]}'
    if prefix == 'model.fc':
        return 'fc'
    raise ValueError(f'Unknown draft runtime namespace: {prefix}')


@register_quantization_config('b70_dflash_w4')
class DraftW4Config(QuantizationConfig):
    def __init__(self, inventory):
        super().__init__()
        self.module_inventory = inventory
        expected = {'fc'} | {f'layers.{i}.{part}' for i in range(5) for part in
            ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
             'mlp.gate_proj','mlp.up_proj','mlp.down_proj']}
        assert set(inventory) == expected
        for key, record in inventory.items():
            assert record['role']=='draft' and record['checkpoint_module']==key
            assert record['bits']==4 and record['group_size']==128 and record['symmetric'] is True
            assert record['zero_point']==8 and record['shape'][1]%128==0

    @staticmethod
    def get_name(): return 'b70_dflash_w4'
    @staticmethod
    def get_supported_act_dtypes(): return [torch.float16]
    @classmethod
    def get_min_capability(cls): return 0
    @staticmethod
    def get_config_filenames(): return []
    @classmethod
    def from_config(cls, config):
        assert config['bits']==4 and config['group_size']==128 and config['symmetric'] is True
        return cls(config['module_inventory'])

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            # Explicitly excluded small projections; no fallback for an
            # unrecognized linear or an inventory mismatch.
            if re.fullmatch(r'model\.layers\.(6[4-8])\.(attention_conv|mlp_conv)\.kernel_projection', prefix) or prefix == 'model.hidden_projection':
                return UnquantizedLinearMethod()
            key = local_key(prefix)
            base, _, leaf = key.rpartition('.')
            members = [f'{base}.{p}' if base else p for p in FUSED.get(leaf, [leaf])]
            assert all(member in self.module_inventory for member in members), members
            return DraftW4Linear(self, prefix, members)
        return None


class DraftW4Linear(LinearMethodBase):
    def __init__(self, config, prefix, members):
        self.config, self.prefix, self.members = config, prefix, members

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        assert get_tensor_model_parallel_world_size()==1
        k, sizes = input_size_per_partition, output_partition_sizes
        n = sum(sizes)
        assert k==input_size and k%128==0 and params_dtype==torch.float16
        assert len(sizes)==len(self.members)
        for member, size in zip(self.members, sizes):
            assert self.config.module_inventory[member]['shape']==[size, k], (member, size, k)
        layer.draft_w4_sizes = list(sizes)
        layer.draft_w4_members = list(self.members)
        layer.draft_w4_loaded = {'qweight': set(), 'scales': set()}
        owner = weakref.ref(layer)

        def loader(name):
            def load(parameter, value, shard_id=None):
                module = owner(); assert module is not None
                if shard_id is None:
                    shards = list(range(len(sizes)))
                elif isinstance(shard_id, str):
                    shards = [QKV_IDS[shard_id]]
                elif isinstance(shard_id, int):
                    shards = [shard_id]
                else:
                    raise ValueError(f'Unsupported W4 shard: {shard_id}')
                assert shards == list(range(shards[0], shards[-1]+1))
                assert not (set(shards)&module.draft_w4_loaded[name]), (self.prefix, name, shards)
                start, end = sum(sizes[:shards[0]]), sum(sizes[:shards[-1]+1])
                expected = (k//8 if name=='qweight' else k//128, end-start)
                dtype = torch.int32 if name=='qweight' else torch.float16
                assert value.shape==expected and value.dtype==dtype, (self.prefix,name,value.shape,value.dtype,expected)
                if name=='scales':
                    assert bool(torch.isfinite(value).all()) and bool((value>0).all())
                parameter.data[:, start:end].copy_(value)
                module.draft_w4_loaded[name].update(shards)
            return load

        for name, tensor in [('qweight', torch.empty(k//8, n, dtype=torch.int32)),
                             ('scales', torch.empty(k//128, n, dtype=torch.float16))]:
            parameter = Parameter(tensor, requires_grad=False)
            parameter.weight_loader = loader(name)
            layer.register_parameter(name, parameter)

    def process_weights_after_loading(self, layer):
        expected = set(range(len(self.members)))
        assert all(value==expected for value in layer.draft_w4_loaded.values()), (self.prefix,layer.draft_w4_loaded)
        # Exact oneDNN NT layout, preserving packed coefficients. No dense copy.
        layer.qweight = Parameter(layer.qweight.data.t().contiguous().t(), requires_grad=False)
        layer.scales = Parameter(layer.scales.data.contiguous(), requires_grad=False)
        layer.register_buffer('draft_w4_zero', torch.tensor([8], dtype=torch.int8, device=layer.qweight.device))
        if os.environ.get('B70_DFLASH_KV_ONLY')=='1' and self.prefix.endswith('.self_attn.qkv_proj'):
            q_size=layer.draft_w4_sizes[0]
            # Packed NT columns alias the same weight; scales are a small
            # contiguous buffer required by the oneDNN ABI.
            layer.register_buffer('draft_w4_kv_weight',layer.qweight.data[:,q_size:])
            layer.register_buffer('draft_w4_kv_scales',layer.scales.data[:,q_size:].contiguous())
        layer.draft_w4_loader_report = dict(role='draft', runtime_prefix=self.prefix,
            checkpoint_members=self.members, bits=4, group_size=128, symmetric=True,
            input_dtype='float16', packed_shape=list(layer.qweight.shape),
            packed_stride=list(layer.qweight.stride()), scales_shape=list(layer.scales.shape),
            dense_weight_copy=False, backend='_xpu_C.int4_gemm_w4a16')

    def apply(self, layer, x, bias=None):
        shape = x.shape[:-1]+(layer.qweight.shape[1],)
        assert x.dtype==torch.float16
        return torch.ops._xpu_C.int4_gemm_w4a16(x.reshape(-1, x.shape[-1]),
            layer.qweight, bias, layer.scales, layer.draft_w4_zero, 128, None).reshape(shape)

    def apply_context_kv(self, layer, x):
        assert x.dtype==torch.float16 and layer.bias is None
        assert self.prefix.endswith('.self_attn.qkv_proj')
        weight=layer.draft_w4_kv_weight
        assert weight.untyped_storage().data_ptr()==layer.qweight.untyped_storage().data_ptr()
        return torch.ops._xpu_C.int4_gemm_w4a16(x.reshape(-1,x.shape[-1]),
            weight,None,layer.draft_w4_kv_scales,layer.draft_w4_zero,128,None)


def audit_draft(runner):
    draft = runner.speculator.model
    assert isinstance(draft.model.quant_config, DraftW4Config)
    expected = set(draft.model.quant_config.module_inventory)
    reports = [module.draft_w4_loader_report for module in draft.modules()
               if hasattr(module, 'draft_w4_loader_report')]
    actual = [name for report in reports for name in report['checkpoint_members']]
    assert len(actual)==len(set(actual))==36 and set(actual)==expected
    assert len(reports)==21
    assert draft.model._fused_kv_weight is None, 'Quantized draft must not keep dense context KV weights'
    for name, parameter in draft.named_parameters():
        if name.startswith(('lm_head.', 'model.embed_tokens.')):
            continue
        assert parameter.dtype in (torch.int32, torch.float16), (name, parameter.dtype)
    result = dict(status='PASS_DRAFT_W4_ROLE_INVENTORY', role='draft',
                  expected_checkpoint_members=36, loaded_runtime_modules=21, modules=reports,
                  runtime_attention_layer_names=draft.get_draft_kv_cache_layer_names())
    return result
