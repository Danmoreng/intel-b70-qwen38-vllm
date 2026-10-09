FROM sha256:b0b9244420a2586f8328b14aeff21d94673632ad7342032a285f3b877f2de49b
COPY draft_w4.py /opt/exl3xpu/exl3xpu/draft_w4.py
COPY patch_registration.py /opt/dflash-w4/patch_registration.py
RUN python /opt/dflash-w4/patch_registration.py /opt/exl3xpu/exl3xpu/vllm_plugin.py
ENV EXL3_DFLASH_W4=1
