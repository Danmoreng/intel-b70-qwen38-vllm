"""Weights-free diagnostic of the legacy V1 recovered-token kernel.

Known sparse distributions make every zero-probability selection invalid.
CPU argmax independently checks the supplied probability/noise tensors.
This diagnostic does not qualify a model or change its sampling settings.
The actual production/candidate V2 Model Runner uses a different sampler;
passing this probe does not cover that active path. CPU input inspection
also synchronizes before this kernel, excluding an async-pipeline claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import triton
from vllm.v1.sample.rejection_sampler import _sample_recovered_tokens

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
report = dict(status='RUNNING_SAMPLER_DIAGNOSTIC', torch=torch.__version__,
              torch_git=torch.version.git_version, triton=triton.__version__,
              device=torch.xpu.get_device_name(0), cases=[],
              active_v2_path_covered=False,
              sampler_module=_sample_recovered_tokens.__module__,
              runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scope='No model or weights. Legacy V1 kernel, not the active V2 runner; CPU-synchronized synthetic inputs. Not release qualification or async-pipeline coverage.')
a.out.parent.mkdir(parents=True, exist_ok=True)


def save():
    a.out.write_text(json.dumps(report, indent=2) + '\n')


save()
try:
    vocab = 248320
    for rows in (1, 3, 4):
        cu = torch.tensor([rows], device='xpu', dtype=torch.int32)
        draft = torch.zeros(rows, device='xpu', dtype=torch.int64)
        for kind in ('one-hot', 'sparse20'):
            probabilities = torch.zeros((rows, vocab), dtype=torch.float32)
            for row in range(rows):
                indices = [12345 + row * 8192] if kind == 'one-hot' else [2048 + row * 19 + n * 11003 for n in range(20)]
                values = torch.arange(1, len(indices) + 1, dtype=torch.float32)
                probabilities[row, indices] = values / values.sum()
            gpu_probs = probabilities.to('xpu')
            for seed in (None, 202611119, 202611120, 1337):
                if seed is None:
                    q = torch.ones((1, vocab), device='xpu', dtype=torch.float32)
                else:
                    generator = torch.Generator(device='xpu').manual_seed(seed)
                    torch.empty((rows,), device='xpu', dtype=torch.float64).uniform_(generator=generator)
                    q = torch.empty((1, vocab), device='xpu', dtype=torch.float32).exponential_(generator=generator)
                inv_q = q.reciprocal()
                q_cpu, inv_cpu = q.cpu(), inv_q.cpu()
                valid_noise = bool(torch.isfinite(q_cpu).all() and (q_cpu > 0).all() and torch.isfinite(inv_cpu).all())
                expected = (probabilities * inv_cpu).argmax(dim=-1)
                actual = torch.full_like(draft, -777)
                _sample_recovered_tokens(actual, cu, draft, None, gpu_probs, inv_q,
                                         max_spec_len=rows)
                got = actual.cpu()
                positive = probabilities[torch.arange(rows), got.long()] > 0 if bool(((got >= 0) & (got < vocab)).all()) else torch.zeros(rows, dtype=torch.bool)
                ok = valid_noise and torch.equal(got.long(), expected) and bool(positive.all())
                report['cases'].append(dict(rows=rows, distribution=kind, seed=seed,
                    valid_exponential_noise=valid_noise, expected=expected.tolist(),
                    actual=got.tolist(), selected_positive_probability=positive.tolist(), passed=ok))
                save()
                print('SAMPLER', rows, kind, seed, 'PASS' if ok else 'FAIL', got.tolist(), expected.tolist(), flush=True)
    report['status'] = 'PASS_SYNTHETIC_SAMPLER_DIAGNOSTIC' if all(c['passed'] for c in report['cases']) else 'FAIL_SYNTHETIC_SAMPLER_DIAGNOSTIC'
except BaseException as exc:
    report.update(status='FAILED_DIAGNOSTIC_EXECUTION', error=repr(exc))
    raise
finally:
    save()
raise SystemExit(0 if report['status'] == 'PASS_SYNTHETIC_SAMPLER_DIAGNOSTIC' else 2)
