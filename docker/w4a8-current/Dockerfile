FROM local/qwen38-b70-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4

COPY w4a8-current/patch_gptq_w4a8_prefill.py /opt/b70-w4a8/
RUN python /opt/b70-w4a8/patch_gptq_w4a8_prefill.py \
 && rm -rf /opt/b70-w4a8
