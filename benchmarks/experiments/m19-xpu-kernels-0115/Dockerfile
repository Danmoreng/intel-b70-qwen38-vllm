FROM local/qwen38-b70-vllm:vllm-0.30.0-20260923

USER root
COPY vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl /tmp/vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl
RUN echo '4ec262f7afdd07c62defc8286dc9f357569627b996c09b410f7972b653696fe7  /tmp/vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl' | sha256sum -c - \
 && /opt/venv/bin/python -m pip install --no-deps --force-reinstall /tmp/vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl \
 && /opt/venv/bin/python -c "import importlib.metadata as m; assert m.version('vllm-xpu-kernels') == '0.1.15.4'" \
 && rm /tmp/vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl

ENTRYPOINT ["vllm", "serve"]
