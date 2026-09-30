"""Frozen real prefill operands: fused current INT8 reference versus opaque dispatch."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from vllm._xpu_ops import xpu_ops
from vllm_xpu_kernels import _xpu_C
import b70_gptq_row_dispatch


def reference(x, packed, scales, zero, group):
    quant, factor, origin = xpu_ops.dynamic_per_token_int8_quant_ref(x, True, 8)
    out = torch.ops._xpu_C.int4_gemm_w4a8(quant, factor, origin, packed.t(), scales, zero, group, None, None).to(x.dtype)
    return out, quant, factor, origin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operands', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compiled = torch.compile(reference, fullgraph=True, dynamic=True)
    rows = []
    for path in sorted(args.operands.glob('prefill-*.pt')):
        data = torch.load(path, weights_only=True)
        packed, scales, zero = (data[name].to('xpu') for name in ('weight', 'scales', 'zero'))
        packed = packed.t()
        for m in (512, 513, data['input'].shape[0]):
            x = data['input'][:m].to('xpu')
            ref, quant, factor, origin = compiled(x, packed, scales, zero, data['group_size'])
            eager = reference(x, packed, scales, zero, data['group_size'])
            fixed = torch.ops.b70_linear.gptq_by_rows(x, packed, scales, zero, data['group_size'], 512, None)
            torch.xpu.synchronize()
            gate = {'file': path.name, 'm': m,
                    'operand_sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
                    'quant_bitwise': torch.equal(quant, eager[1]),
                    'factor_bitwise': torch.equal(factor, eager[2]),
                    'origin_bitwise': torch.equal(origin, eager[3]),
                    'output_bitwise': torch.equal(ref, fixed),
                    'output_max_abs': float((ref.float() - fixed.float()).abs().max())}
            rows.append(gate)
            args.output.write_text(json.dumps({'scope': 'frozen real large-M operands, fused INT8 reference vs row dispatch', 'rows': rows}, indent=2))
            if not all(gate[key] for key in ('quant_bitwise', 'factor_bitwise', 'origin_bitwise', 'output_bitwise')):
                raise RuntimeError('prefill numeric dispatch gate failed: ' + repr(gate))
    if not rows:
        raise RuntimeError('no real prefill operands captured')


if __name__ == '__main__':
    main()
