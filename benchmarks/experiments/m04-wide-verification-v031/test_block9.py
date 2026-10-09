#!/usr/bin/env python3
"""Nine-token DFlash FP16 convolution boundaries, compile and graph replay."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from vllm.config import VllmConfig, set_current_vllm_config

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(731)
config = VllmConfig()
config.compilation_config.mode = 0
config.kernel_config.set_platform_defaults(config)
result = dict(status='RUNNING', checks=[], device='xpu', block_size=9)
with set_current_vllm_config(config), config.kernel_config.ir_op_priority.set_priority():
    import vllm.model_executor.models.qwen3_dflash2 as module
    source_sha = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert source_sha == '34da1d98d4f89c0b32ee26be95555c5098440d7d2d38931f6f913483dcac3e3e'
    result['dflash_source_sha256'] = source_sha
    conv = module._grouped_conv
    def reference(x, delta, base):
        value = torch.zeros_like(x, dtype=torch.float32)
        groups = torch.arange(5120, device='xpu') // 16
        for row in range(x.shape[0]):
            for tap in range(min(2, row % 9 + 1)):
                value[row] += (base[tap].float() + delta[row, tap, groups].float()) * x[row-tap].float()
        return (value / 16).half()
    for rows in [0, 1, 9, 36]:
        x = (torch.randn(rows, 10240, device='xpu') * 300).half()[:, ::2]
        delta = torch.randn(rows, 2, 2, 320, device='xpu').half()[:, 1]
        base = (torch.randn(2, 5120, device='xpu') * 2).half()
        if rows:
            x[-1, 0] = 60000
            base[0, 0] = 2
            delta[-1, 0, 0] = 0
        actual = conv(x, delta, base, 9, 320, 16, 2, 1/16)
        expected = reference(x, delta, base)
        assert actual.dtype == torch.float16 and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=.002, atol=.0625)
        result['checks'].append(dict(rows=rows, finite=True, fp32_reference_pass=True))
    x = torch.randn(36, 5120, device='xpu', dtype=torch.float16)
    delta = torch.randn(36, 2, 320, device='xpu', dtype=torch.float16)
    base = torch.randn(2, 5120, device='xpu', dtype=torch.float16)
    def call(x, delta, base):
        return conv(x, delta, base, 9, 320, 16, 2, 1/16)
    compiled = torch.compile(call, fullgraph=True)
    torch.testing.assert_close(compiled(x, delta, base), call(x, delta, base), rtol=0, atol=0)
    for _ in range(3): compiled(x, delta, base)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph): output = compiled(x, delta, base)
    x.mul_(2)
    graph.replay()
    torch.testing.assert_close(output, call(x, delta, base), rtol=0, atol=0)
    result['checks'].append(dict(compile_and_graph_replay='PASS', dtype='torch.float16'))
result['status'] = 'PASS_NEW_BLOCK9_FP16_PRIMITIVES'
args.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
