"""Exercise the installed model-route device guard without GPU pointers."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from exl3xpu import attention_dispatch as dispatch
from exl3xpu import shared_kv_verify as verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    q = torch.empty((64, 24, 256), dtype=torch.float16, device='cpu')
    k = torch.empty((1, 64, 4, 256), dtype=torch.float8_e4m3fn, device='cpu')
    scale = torch.ones(1, dtype=torch.float32, device='cpu')
    arguments = dict(q=q, k=k, v=torch.empty_like(k), out=torch.empty_like(q),
                     cu_seqlens_q=torch.tensor([0, 64], dtype=torch.int32),
                     seqused_k=torch.tensor([4096], dtype=torch.int32),
                     block_table=torch.zeros((1, 64), dtype=torch.int32),
                     k_descale=scale, v_descale=scale,
                     causal=True, softmax_scale=.0625)
    assert all(t.device.type == 'cpu' for t in arguments.values() if isinstance(t, torch.Tensor))
    context = dispatch.Context(True, plan=SimpleNamespace(tokens=64, requests=1, segments=[]), scales=(scale, scale))
    assert dispatch.supported_features(arguments)
    assert not dispatch.prefill_eligible(arguments, context, capturing=False)
    assert not verify.eligible(arguments)
    assert dispatch.select_route(arguments, context, capturing=False,
                                 native_op=object(), m04=True, onednn=True) == 'native'
    paths = [Path(dispatch.__file__), Path(verify.__file__)]
    report = dict(status='PASS_CPU_OPERANDS_EXCLUDED_FROM_CUSTOM_ATTENTION_ROUTES',
                  source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  cpu_prefill_excluded=True, cpu_m04_excluded=True,
                  any_custom_native_op_called=False, any_gpu_tensor_allocated=False,
                  scope='Actual installed Python predicates executed on real CPU tensors in a container with no GPU device mounts. This tests model-route exclusion, not CPU support or rejection by the direct XPU-native API, whose operand precondition remains XPU. No wrong-device pointer is supplied to a native operation.')
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'])


if __name__ == '__main__':
    main()
