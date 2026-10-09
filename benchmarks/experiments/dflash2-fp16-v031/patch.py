#!/usr/bin/env python3
"""Guarded DFlash2 FP16 residual representation patch; no checkpoint changes."""
import ast
import hashlib
import json
from pathlib import Path
import sys

EXPECTED = '11ac6ba8c6db702375970e68cd042b98783a5c43b87f9bfa27e88463316bf499'

def patched(source):
    assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED, 'unsupported upstream source'
    def replace(old, new, count=1):
        nonlocal source
        assert source.count(old) == count, (old, source.count(old), count)
        source = source.replace(old, new)
    replace('    ELEMENT_BLOCK: tl.constexpr,\n', '    ELEMENT_BLOCK: tl.constexpr,\n    OUTPUT_SCALE: tl.constexpr,\n')
    replace('        accumulator,\n        mask=mask,', '        accumulator * OUTPUT_SCALE,\n        mask=mask,')
    replace('    group_size: int,\n) -> torch.Tensor:', '    group_size: int,\n    output_scale: float = 1.0,\n) -> torch.Tensor:', 3)
    replace('    num_rows, num_channels = x.shape',
            '    # Kernel channel/group offsets require unit inner strides.\n    # Normal model tensors already satisfy this; canonicalize other views.\n    x = x.contiguous()\n    delta = delta.contiguous()\n    base = base.contiguous()\n    num_rows, num_channels = x.shape')
    replace('        ELEMENT_BLOCK=element_block,\n', '        ELEMENT_BLOCK=element_block,\n        OUTPUT_SCALE=output_scale,\n')
    replace('return torch.ops.vllm.dflash2_grouped_conv(x, delta, base, block_size, group_size)',
            'return torch.ops.vllm.dflash2_grouped_conv(\n        x, delta, base, block_size, group_size, output_scale\n    )')
    replace('    taps: int,\n) -> torch.Tensor:', '    taps: int,\n    output_scale: float = 1.0,\n) -> torch.Tensor:')
    replace('    if hidden_states.is_cuda:\n        return dflash2_grouped_conv(hidden_states, delta, base, block_size, group_size)',
            '    if hidden_states.is_cuda or (hidden_states.is_xpu and hidden_states.dtype == torch.float16):\n        return dflash2_grouped_conv(\n            hidden_states, delta, base, block_size, group_size, output_scale\n        )')
    replace('    blocks = hidden_states.unflatten(-1, (num_groups, group_size))',
            '    # FP16 convolution must scale before the final narrow store.\n    # CPU fallback is also the reference for the FP32-accumulating device kernel.\n    accumulator_dtype = torch.float32 if hidden_states.dtype == torch.float16 else hidden_states.dtype\n    blocks = hidden_states.to(accumulator_dtype).unflatten(-1, (num_groups, group_size))\n    delta = delta.to(accumulator_dtype)\n    base = base.to(accumulator_dtype)')
    replace('    return output.flatten(-2)', '    return (output.flatten(-2) * output_scale).to(hidden_states.dtype)')
    replace('self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int\n',
            'self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int,\n        output_scale: float = 1.0,\n')
    replace('            self.taps,\n', '            self.taps,\n            output_scale,\n')
    replace('self, hidden_states: torch.Tensor, coefficients: torch.Tensor\n',
            'self, hidden_states: torch.Tensor, coefficients: torch.Tensor,\n        output_scale: float = 1.0,\n')
    replace('return self._convolve(hidden_states, coefficients, 1)',
            'return self._convolve(hidden_states, coefficients, 1, output_scale)')
    replace('        draft_config = config.dflash_config\n',
            '        # Store the whole decoder residual stream in units of 1/16.\n        # RMSNorm(a*x, a*a*eps) == RMSNorm(x, eps); branch inputs stay unchanged.\n        self.residual_scale = 1.0 / 16 if vllm_config.model_config.dtype == torch.float16 else 1.0\n        self.input_layernorm.variance_epsilon *= self.residual_scale ** 2\n        self.post_attention_layernorm.variance_epsilon *= self.residual_scale ** 2\n        draft_config = config.dflash_config\n')
    replace('            residual = hidden_states\n            hidden_states = self.input_layernorm(hidden_states)',
            '            residual = hidden_states * self.residual_scale\n            hidden_states = self.input_layernorm(residual)')
    replace('self.attention_conv.finish(hidden_states, coefficients)',
            'self.attention_conv.finish(hidden_states, coefficients, self.residual_scale)')
    replace('self.mlp_conv.finish(hidden_states, coefficients)',
            'self.mlp_conv.finish(hidden_states, coefficients, self.residual_scale)')
    replace('        self.input_embedding_scale = float(',
            '        if vllm_config.model_config.dtype == torch.float16:\n            self.norm.variance_epsilon *= (1.0 / 16) ** 2\n        self.input_embedding_scale = float(')
    ast.parse(source)
    return source

if __name__ == '__main__':
    target = Path(sys.argv[1])
    original = target.read_text()
    result = patched(original)
    target.write_text(result)
    print(json.dumps(dict(upstream_sha256=EXPECTED, patched_sha256=hashlib.sha256(result.encode()).hexdigest())))
