FROM sha256:194a8786f12065e65647d914ddc7fdc136bafbeec73f476730ef1324e4d2c7ec
COPY patch.py /opt/dflash2-fp16/patch.py
RUN python /opt/dflash2-fp16/patch.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash2.py
