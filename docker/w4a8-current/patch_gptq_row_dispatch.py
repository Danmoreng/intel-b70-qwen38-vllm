#!/usr/bin/env python3
"""Upgrade only the pinned GPTQ wrapper; keep weights and the 512-row policy."""
import hashlib
from pathlib import Path

LINEAR = Path('/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/xpu.py')
EXPECTED = '58b0b4fc0972d1bfd2bdf294910fa6bd5841d0c511d1de41ca88d9164b884945'
OLD = '''        if self.config.weight_type == scalar_types.uint4b8:
            if reshaped_x.shape[0] < self._b70_w4a8_min_tokens:
                return XPUwNa16LinearKernel.apply_weights(self, layer, x, bias)
            from vllm._xpu_ops import xpu_ops as ops

            w_q, w_s, w_zp = self._get_weight_params(layer)
            quant_x, x_scale, x_zero = ops.dynamic_per_token_int8_quant_ref(
                reshaped_x, True, 8
            )
            out = torch.ops._xpu_C.int4_gemm_w4a8(
                quant_x, x_scale, x_zero, w_q.t(), w_s, w_zp,
                self.config.group_size, None, bias,
            )
            return out.to(x.dtype)'''
NEW = '''        if self.config.weight_type == scalar_types.uint4b8:
            w_q, w_s, w_zp = self._get_weight_params(layer)
            return torch.ops.b70_linear.gptq_by_rows(
                reshaped_x, w_q, w_s, w_zp, self.config.group_size,
                self._b70_w4a8_min_tokens, bias,
            )'''


def main():
    source = LINEAR.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED or source.count(OLD) != 1:
        raise RuntimeError('pinned GPTQ source differs')
    source = source.replace('\nimport torch\n',
        '\nimport torch\n\nimport b70_gptq_row_dispatch  # registers b70_linear::gptq_by_rows\n', 1)
    source = source.replace(OLD, NEW, 1)
    compile(source, str(LINEAR), 'exec')
    LINEAR.write_text(source)
    linear_hash = hashlib.sha256(source.encode()).hexdigest()
    helper = Path('/opt/venv/lib/python3.12/site-packages/b70_gptq_row_dispatch.py')
    helper_hash = hashlib.sha256(helper.read_bytes()).hexdigest()
    verifier = Path('/opt/b70/verify_production.py')
    check = verifier.read_text()
    if check.count(EXPECTED) != 1:
        raise RuntimeError('pinned verifier differs')
    check = check.replace(EXPECTED, linear_hash)
    anchor = '        XPU_LINEAR: EXPECTED_LINEAR,\n'
    if check.count(anchor) != 1:
        raise RuntimeError('verifier files anchor differs')
    check = check.replace(anchor, anchor + f'        Path({str(helper)!r}): {helper_hash!r},\n')
    compile(check, str(verifier), 'exec')
    verifier.write_text(check)
    print('B70_ROW_DISPATCH_PATCH', {'linear_sha256': linear_hash, 'helper_sha256': helper_hash}, flush=True)


if __name__ == '__main__':
    main()
