FROM local/qwen38-b70-vllm:q128-m04-196k-20260915
COPY benchmarks/experiments/m05-int2-draft-head/capture/b70_capture_hidden.py /opt/venv/lib/python3.12/site-packages/b70_capture_hidden.py
COPY benchmarks/experiments/m05-int2-draft-head/capture/patch_capture.py /tmp/patch_capture.py
RUN python /tmp/patch_capture.py && rm /tmp/patch_capture.py
