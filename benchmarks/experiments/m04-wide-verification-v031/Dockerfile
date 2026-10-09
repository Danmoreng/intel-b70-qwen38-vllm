FROM sha256:a24e00efd83df6fb944137223c9da0747c48cc105641aae62e021340b4486788
COPY build/m04.so /opt/exl3xpu/m04/m04.so
COPY patch_wrapper.py /opt/m04-wide/patch_wrapper.py
RUN python /opt/m04-wide/patch_wrapper.py /opt/exl3xpu/exl3xpu/shared_kv_verify.py models/qwen3.8-27b-exl3-4.00bpw/migration-target-optimized.yaml
LABEL org.local.b70.experiment="M04 q2..9; unqualified DFlash speed experiment"
