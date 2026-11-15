FROM sha256:dc81825313d587ca22eaeeceb26808a8eed63cfb84b9dde22599223b8dab59d1 AS reference
FROM sha256:b37eb3926bade76db626a261b98de6570036fd59d81a3a06e6ab780a87f0918d
RUN rm -rf /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels-0.1.17.1.dist-info
COPY --from=reference /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
COPY --from=reference /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels-0.1.15.4.dist-info /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels-0.1.15.4.dist-info
RUN python -c 'import torch,importlib.metadata as m; import vllm_xpu_kernels; assert torch.__version__=="2.15.0+xpu"; assert m.version("vllm-xpu-kernels")=="0.1.15.4"; print("CPU_IMPORT_ONLY_NOT_GPU_QUALIFIED")'
LABEL org.local.b70.release-status="DIAGNOSTIC_ONLY_UNQUALIFIED"
