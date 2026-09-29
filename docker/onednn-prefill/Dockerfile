FROM local/b70-qwen38-vllm:w4a8-current-20260929

ARG B70_POLICY_SHA256
LABEL org.local.b70.policy.sha256=$B70_POLICY_SHA256
ENV B70_POLICY_SHA256=$B70_POLICY_SHA256 \
    B70_GPTQ_W4A8_PREFILL=1 \
    B70_GPTQ_W4A8_MIN_TOKENS=512 \
    B70_ONEDNN_PREFILL=1 \
    B70_ONEDNN_MIXED_ROUTE=1 \
    B70_ONEDNN_MIN_KV=16384 \
    B70_ONEDNN_MAX_KV=196608

COPY docker/onednn-prefill/stage/libdnnl.so.3.13 /opt/b70/libdnnl.so.3
COPY docker/onednn-prefill/stage/native_sdpa.so /opt/b70/native_sdpa.so
COPY docker/m04/b70_attention.py /opt/venv/lib/python3.12/site-packages/b70_attention_base.py
COPY docker/onednn-prefill/b70_attention.py /opt/venv/lib/python3.12/site-packages/b70_attention.py
COPY docker/onednn-prefill/verify_production.py /opt/b70/verify_production.py
COPY config/production_policy.json /opt/b70/production_policy.json
COPY config/production_policy.sha256 /opt/b70/production_policy.sha256

ENV LD_LIBRARY_PATH=/opt/b70:/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib

RUN python /opt/b70/verify_production.py --files-only \
 && python -c "import torch; torch.ops.load_library('/opt/b70/native_sdpa.so'); assert hasattr(torch.ops.b70_sdpa_probe, 'forward')"

ENTRYPOINT ["python", "/opt/b70/verify_production.py", "--serve"]
