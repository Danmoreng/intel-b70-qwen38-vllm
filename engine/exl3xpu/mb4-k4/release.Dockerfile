FROM sha256:dfd1514f44b59b634eb20148bd4343a8807c3991bdcf46c11a21aa06eae583bb
COPY source/ /opt/exl3xpu/native-matrix-source/
COPY verify_native_artifact.py /opt/exl3xpu/scripts/verify_native_artifact.py
LABEL org.local.b70.policy.sha256="ec957c94a9d2f51511891387d271c6acd4b7dd356186fa11bb3d3852cae80463" org.local.b70.release-status="PRODUCTION" org.opencontainers.image.source="https://github.com/Danmoreng/intel-b70-qwen38-vllm"
ENV B70_POLICY_SHA256=ec957c94a9d2f51511891387d271c6acd4b7dd356186fa11bb3d3852cae80463
