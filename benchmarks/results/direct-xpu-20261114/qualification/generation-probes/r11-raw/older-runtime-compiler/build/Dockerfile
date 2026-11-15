FROM sha256:b37eb3926bade76db626a261b98de6570036fd59d81a3a06e6ab780a87f0918d AS selected
FROM sha256:955c847e06c54d42a0e671e933e7f1a0611837938f41feefff041d94799f5efa
COPY --from=selected /opt/venv/lib/libpti_view.so.1.2.1 /opt/venv/lib/libpti_view.so.1.2.1
COPY --from=selected /opt/venv/lib/pti/libpti.so /opt/venv/lib/pti/libpti.so
RUN rm -rf /opt/venv/lib/python3.12/site-packages/transformers /opt/venv/lib/python3.12/site-packages/transformers-5.23.0.dist-info
COPY --from=selected /opt/venv/lib/python3.12/site-packages/transformers /opt/venv/lib/python3.12/site-packages/transformers
COPY --from=selected /opt/venv/lib/python3.12/site-packages/transformers-5.22.2.dist-info /opt/venv/lib/python3.12/site-packages/transformers-5.22.2.dist-info
RUN python -c 'import importlib.metadata as m; assert m.version("torch")=="2.15.0+xpu"; assert m.version("intel-sycl-rt")=="2026.1.2"; assert m.version("transformers")=="5.22.2"; print("CPU_VERSION_ONLY_NOT_GPU_QUALIFIED")'
LABEL org.local.b70.release-status="DIAGNOSTIC_ONLY_UNQUALIFIED"
