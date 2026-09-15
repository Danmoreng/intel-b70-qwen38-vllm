ARG BASE_IMAGE=local/qwen38-b70-vllm:q128-196k-20260914
FROM ${BASE_IMAGE}
COPY b70_ops/build/b70_ops.so /opt/b70_ops/b70_ops.so
COPY b70_ops/python/b70_ops /opt/venv/lib/python3.12/site-packages/b70_ops
COPY docker/direct-output/patch_attention_import.py /tmp/patch_attention_import.py
RUN python /tmp/patch_attention_import.py && rm /tmp/patch_attention_import.py
