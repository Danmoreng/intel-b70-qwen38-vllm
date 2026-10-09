#!/usr/bin/env python3
"""Independent FP32 reference gates for the patched FP16 convolution/residual."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from vllm.config import VllmConfig, set_current_vllm_config

p = argparse.ArgumentParser()
p.add_argument('--device', choices=['cpu', 'xpu'], required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
torch.manual_seed(731)
SCALE = 1 / 16
EPS = 1e-6
results = dict(device=a.device, scale=SCALE, torch_version=torch.__version__, checks=[])
config = VllmConfig()
config.compilation_config.mode = 0
if a.device == "xpu":
    config.kernel_config.set_platform_defaults(config)
with set_current_vllm_config(config), config.kernel_config.ir_op_priority.set_priority():
    from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv
    from vllm.model_executor.layers.layernorm import RMSNorm
    import vllm.model_executor.models.qwen3_dflash2 as module
    results['patched_source_sha256'] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    def ref_conv(x, delta, base, block, group):
        # Independent scalar tap loop; each request is its own causal convolution.
        y = torch.zeros_like(x, dtype=torch.float32)
        channels = torch.arange(x.shape[1], device=x.device) // group
        for row in range(x.shape[0]):
            for tap in range(min(base.shape[0], row % block + 1)):
                y[row] += (base[tap].float() + delta[row, tap, channels].float()) * x[row-tap].float()
        return y
    for block in [4, 6, 8]:
        for requests in [1, 4]:
            for rows in [0, 1, block * requests]:
                width, group, taps = 5120, 16, 2
                # Strided x, strided tap/group views exercise real prepare() layout.
                x = (torch.randn(rows, width*2, device=a.device)*300).half()[:, ::2]
                base = (torch.randn(taps, width, device=a.device)*2).half()
                delta = torch.randn(rows, 2, taps, width//group, device=a.device).half()[:, 1]
                if rows:
                    x[-1, 0] = 60000
                    base[0, 0] = 2
                    delta[-1, 0, 0] = 0
                expected = (ref_conv(x, delta, base, block, group)*SCALE).half()
                actual = _grouped_conv(x, delta, base, block, width//group, group, taps, SCALE)
                assert actual.dtype == torch.float16 and torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=.002, atol=.0625)
                results['checks'].append(dict(test='convolution', block=block, requests=requests, rows=rows, max_abs_error=float((actual-expected).abs().max()) if rows else 0))
    # The reported fused-add overflow, not a hypothetical training precision issue.
    x = torch.full((4, 5120), 11424., device=a.device, dtype=torch.float16)
    residual = torch.full_like(x, 65216.)
    old_norm = RMSNorm(5120, eps=EPS, dtype=torch.float16).to(a.device)
    new_norm = RMSNorm(5120, eps=EPS*SCALE*SCALE, dtype=torch.float16).to(a.device)
    old_out, old_residual = old_norm(x.clone(), residual.clone())
    assert not torch.isfinite(old_residual).all(), 'reported overflow not reproduced'
    out, retained = new_norm(x*SCALE, residual*SCALE)
    assert torch.isfinite(out).all() and torch.isfinite(retained).all()
    assert out.dtype == retained.dtype == torch.float16
    total = x.float()+residual.float()
    reference = total*torch.rsqrt(total.square().mean(-1,keepdim=True)+EPS)
    torch.testing.assert_close(out.float(), reference, rtol=.002, atol=.002)
    results['checks'].append(dict(test='reported-residual-overflow', old_residual_finite=False, patched_finite=True, output_max_abs_error=float((out.float()-reference).abs().max())))
    # Epsilon and near-zero cases catch invalid "just divide everything" patches.
    for magnitude in [0, 1e-4, 1e-3, 1., 1000., 60000.]:
        x=(torch.randn(4,5120,device=a.device)*magnitude).clamp(-60000,60000).half()
        expected=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+EPS)
        actual=new_norm(x*SCALE)
        assert actual.dtype == torch.float16 and torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(),expected,rtol=.004,atol=.002)
        results['checks'].append(dict(test='epsilon-sensitive-norm', magnitude=magnitude, max_abs_error=float((actual.float()-expected).abs().max())))
    if a.device == 'xpu':
        x=torch.randn(16,5120,device='xpu',dtype=torch.float16)
        delta=torch.randn(16,2,320,device='xpu',dtype=torch.float16)
        base=torch.randn(2,5120,device='xpu',dtype=torch.float16)
        def conv(x,delta,base):
            return _grouped_conv(x,delta,base,4,320,16,2,SCALE)
        compiled=torch.compile(conv,fullgraph=True)
        expected=conv(x,delta,base)
        actual=compiled(x,delta,base)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        # Graph replay uses the actual custom-op Triton kernel, not a fake result.
        graph=torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            replay=conv(x,delta,base)
        x.copy_(x*2)
        graph.replay()
        torch.testing.assert_close(replay,conv(x,delta,base),rtol=0,atol=0)
        results['checks'].append(dict(test='compile-and-xpu-graph-replay', status='PASS'))
results['status']='PASS_PRIMITIVES_ONLY_NOT_REAL_DRAFT_QUALIFICATION'
a.out.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
