FROM local/qwen38-b70-vllm:q128-m04-196k-20260915
COPY benchmarks/experiments/m06-target-head-int4/candidate/patch_target_head.py /tmp/patch_target_head.py
RUN python /tmp/patch_target_head.py && rm /tmp/patch_target_head.py
ENV B70_TARGET_LMHEAD_INT4=1
