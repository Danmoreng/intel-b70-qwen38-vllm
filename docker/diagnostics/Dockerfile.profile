ARG BASE_IMAGE=local/qwen38-b70-vllm:q128-196k-20260914
FROM ${BASE_IMAGE}

USER root
COPY diagnostics/patch_profile_before_capture.py /tmp/patch_profile_before_capture.py
RUN python /tmp/patch_profile_before_capture.py \
 && rm /tmp/patch_profile_before_capture.py

# The patch is inert unless this diagnostic-only environment flag is enabled.
ENV B70_PROFILE_BEFORE_CAPTURE=0
