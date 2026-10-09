#!/usr/bin/env python3
"""Lossless packed-layout conversion; BF16 dense tensors become FP16 offline."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SOURCE_SHA = 'ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.source/'model.safetensors') == SOURCE_SHA
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.source/'config.json').read_text())
    qconfig = config['quantization_config']
    group = qconfig['config_groups']['group_0']['weights']
    assert group['num_bits']==4 and group['group_size']==128 and group['symmetric'] is True
    assert qconfig['format']=='pack-quantized'
    modules = {}; tensors = {}; conversion = []
    with safe_open(args.source/'model.safetensors', framework='pt', device='cpu') as weights:
        keys = set(weights.keys())
        quantized = {key.removesuffix('.weight_packed') for key in keys if key.endswith('.weight_packed')}
        expected = {'fc'} | {f'layers.{i}.{part}' for i in range(5) for part in
            ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
             'mlp.gate_proj','mlp.up_proj','mlp.down_proj']}
        assert quantized == expected, (quantized-expected, expected-quantized)
        used = set()
        for module in sorted(quantized):
            packed = weights.get_tensor(module+'.weight_packed')
            scale = weights.get_tensor(module+'.weight_scale')
            shape = weights.get_tensor(module+'.weight_shape').tolist()
            n, k = shape
            assert packed.dtype == torch.int32 and list(packed.shape)==[n,k//8]
            assert scale.dtype == torch.float16 and list(scale.shape)==[n,k//128]
            assert bool(torch.isfinite(scale).all()) and bool((scale>0).all())
            assert module+'.weight_zero_point' not in keys and module+'.weight_g_idx' not in keys
            # Compressed-tensors and GPTQ both store sequential offset-binary
            # int4 along K. This only transposes layout; no requantization.
            target_packed = packed.t().contiguous()
            target_scale = scale.t().contiguous()
            assert torch.equal(target_packed.t(), packed)
            assert torch.equal(target_scale.t(), scale)
            tensors[module+'.qweight'] = target_packed
            tensors[module+'.scales'] = target_scale
            used |= {module+'.'+suffix for suffix in ('weight_packed','weight_scale','weight_shape')}
            modules[module] = dict(role='draft', checkpoint_module=module,
                shape=[n,k], bits=4, group_size=128, symmetric=True, zero_point=8,
                source_packed_shape=list(packed.shape), packed_shape=list(target_packed.shape),
                source_packed_sha256=hashlib.sha256(packed.numpy().tobytes()).hexdigest())
        for name in sorted(keys-used):
            value = weights.get_tensor(name)
            assert value.dtype in (torch.float16, torch.bfloat16), (name, value.dtype)
            value = value.to(torch.float16)
            assert bool(torch.isfinite(value).all()), name
            # The published selector tables are bare Parameters. Keep that
            # exact namespace; adding .weight would address a nonexistent child.
            tensors[name] = value.contiguous()
            conversion.append(dict(name=name, output_dtype='float16', shape=list(value.shape)))
    assert len(modules)==36
    config['dtype'] = 'float16'
    config['quantization_config'] = dict(quant_method='b70_dflash_w4', bits=4,
        group_size=128, symmetric=True, module_inventory=modules)
    save_file(tensors, args.output/'model.safetensors', metadata={'format':'pt'})
    (args.output/'model.safetensors').chmod(0o644)
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (args.output/'model.safetensors.index.json').write_text(json.dumps(dict(
        metadata={'total_size':sum(t.numel()*t.element_size() for t in tensors.values())},
        weight_map={name:'model.safetensors' for name in tensors}),indent=2)+'\n')
    result = dict(status='PACKED_LAYOUT_CONVERTED_NOT_RUNTIME_QUALIFIED', source_sha256=SOURCE_SHA,
        source_repository='syvai/Qwen3.8-27B-DFlash2-W4A16',
        source_revision='4d30ec736ffc6b8688dc2ae2b502d9b48bdec279',
        output_sha256=sha(args.output/'model.safetensors'), config_sha256=sha(args.output/'config.json'),
        logical_weight_bytes=sum(t.numel()*t.element_size() for t in tensors.values()),
        dense_fp16=conversion, modules=modules)
    (args.output/'conversion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],result['logical_weight_bytes'],result['output_sha256'],flush=True)


if __name__=='__main__':
    main()
