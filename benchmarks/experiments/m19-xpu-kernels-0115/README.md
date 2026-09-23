# XPU kernels 0.1.15.4 on published vLLM 0.30.0

The production image has `vllm-xpu-kernels 0.1.14.1`, vLLM 0.30.0+xpu and
Torch 2.13.0+xpu. The isolated candidate takes that exact production image and
replaces only the XPU kernel wheel with PyPI version 0.1.15.4. The wheel SHA-256
is pinned in [`Dockerfile`](Dockerfile). The existing Q128/M04 binaries, local
patches, model and serving settings are unchanged.

The wheel can be downloaded into the ignored `runs/wheel/` directory and the
candidate built with:

```bash
python -m pip download --no-deps --dest \
  benchmarks/experiments/m19-xpu-kernels-0115/runs/wheel \
  'vllm-xpu-kernels==0.1.15.4'
docker build --pull=false \
  -f benchmarks/experiments/m19-xpu-kernels-0115/Dockerfile \
  -t local/qwen38-b70-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4 \
  benchmarks/experiments/m19-xpu-kernels-0115/runs/wheel
```

Verify the current production base-image ID is
`sha256:cd6562f03c8328fe60ca69269d0e4175a284859e56525950fbfc021be19d73f3`
before building. [`run_ab.py`](run_ab.py) pins both completed image IDs, uses
the existing exclusive GPU lock and restores the original production service.
It reuses the four frozen Wikipedia prompts from the earlier MTP experiment.
Its run directory, including response text, streams and server logs, is ignored.
See the [tracked result](../../runs/2026-09-23-xpu-kernels-0115/README.md).
