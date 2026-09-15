FROM local/qwen38-b70-vllm:q128-196k-20260914
COPY b70_ops/build/b70_ops.so /opt/b70/m04.so
COPY benchmarks/experiments/m04-shared-kv-verification/candidate/b70_attention.py /opt/venv/lib/python3.12/site-packages/b70_attention.py
