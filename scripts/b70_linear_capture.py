"""Real small-M operands from an explicitly eager offline worker."""
import functools
import os
import sys
from pathlib import Path


def install():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    root = Path('/evidence/linear-operands')
    seen = set()

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func._schema.name == '_xpu_C::int4_gemm_w4a16':
                x, weight, bias, scales, zero, group, retained = args
                m, k = x.shape
                n = weight.shape[1]
                key = (m, k, n)
                small = m in (1, 4, 5, 20) and (k, n) in ((5120, 34816), (17408, 5120), (5120, 5120))
                prefill = m >= 512 and (k, n) == (5120, 34816) and sum(shape[0] >= 512 for shape in seen) < 2
                if (small or prefill) and key not in seen:
                    seen.add(key)
                    root.mkdir(exist_ok=True)
                    torch.save({'input': x.cpu(), 'weight': weight.cpu(),
                                'bias': bias.cpu() if bias is not None else None,
                                'scales': scales.cpu(), 'zero': zero.cpu(),
                                'group_size': group, 'retained': retained,
                                'expected': result.cpu() if small else None,
                                'kind': 'small_linear' if small else 'prefill_quant_gate', 'shape': key,
                                'weight_stride': list(weight.stride()),
                                'source': 'real eager sampled model call; not scored'},
                               root / f'{"m" if small else "prefill-m"}{m}-k{k}-n{n}.pt')
            return result

    for name in ('execute_model', 'sample_tokens'):
        original = getattr(GPUModelRunner, name)

        def wrap(original):
            @functools.wraps(original)
            def call(self, *args, **kwargs):
                if not Path('/evidence/linear-capture-active').exists():
                    return original(self, *args, **kwargs)
                with Capture():
                    return original(self, *args, **kwargs)
            return call

        setattr(GPUModelRunner, name, wrap(original))


if os.environ.get('B70_LINEAR_CAPTURE') == '1' and not any('cpuinfo' in arg for arg in sys.argv):
    install()
