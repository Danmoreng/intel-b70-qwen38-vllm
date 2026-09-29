FROM local/b70-qwen38-vllm:w4a8-current-20260929

COPY onednn-prefill/stage/libdnnl.so.3.13 /opt/b70/libdnnl.so.3
COPY onednn-prefill/stage/native_sdpa.so /opt/b70/native_sdpa.so
COPY m04/b70_attention.py /opt/venv/lib/python3.12/site-packages/b70_attention_base.py
COPY onednn-prefill/b70_attention.py /opt/venv/lib/python3.12/site-packages/b70_attention.py

ENV LD_LIBRARY_PATH=/opt/b70:/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib

RUN python -c "import torch; torch.ops.load_library('/opt/b70/native_sdpa.so'); assert hasattr(torch.ops.b70_sdpa_probe, 'forward')"
