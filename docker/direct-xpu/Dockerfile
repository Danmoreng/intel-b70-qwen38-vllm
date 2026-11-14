# Direct vLLM 0.33 XPU build. No earlier vLLM image or local model image.
FROM vllm/vllm-openai-xpu@sha256:010bdc441c72ee222f459d3b688bb41317525f1bd7223afe6218490b0c526832 AS upstream
FROM upstream AS runtime-base
USER root
# Preserve the base's newer CCL/runtime components. Apply the selected
# Torch/Triton/runtime upgrades explicitly, without dependency downgrades.
COPY artifacts/ /tmp/b70-artifacts/
RUN pip install --no-cache-dir --no-deps /tmp/b70-artifacts/torch-*.whl \
 && pip install --no-cache-dir --no-deps /tmp/b70-artifacts/vllm_xpu_kernels-*.whl \
 && pip install --no-cache-dir --upgrade triton-xpu==3.9.0 \
 && pip install --no-cache-dir --upgrade intel-sycl-rt==2026.2.0 dpcpp-cpp-rt==2026.2.0 intel-pti==1.2.1 \
 && pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/xpu torchvision==0.29.2+xpu \
 && rm -rf /tmp/b70-artifacts
COPY direct/requirements.txt /tmp/b70-requirements.txt
RUN pip install --no-cache-dir --no-deps -r /tmp/b70-requirements.txt
# Latest Intel userspace, independently SHA-verified; host kernel unchanged.
ADD https://github.com/intel/compute-runtime/releases/download/26.40.40196.8/intel-ocloc_26.40.40196.8-0_amd64.deb /tmp/intel-runtime/intel-ocloc_26.40.40196.8-0_amd64.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.40.40196.8/intel-opencl-icd_26.40.40196.8-0_amd64.deb /tmp/intel-runtime/intel-opencl-icd_26.40.40196.8-0_amd64.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.40.40196.8/libigdgmm12_22.10.2_amd64.deb /tmp/intel-runtime/libigdgmm12_22.10.2_amd64.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.40.40196.8/libze-intel-gpu1_26.40.40196.8-0_amd64.deb /tmp/intel-runtime/libze-intel-gpu1_26.40.40196.8-0_amd64.deb
ADD https://github.com/intel/intel-graphics-compiler/releases/download/v2.42.6/intel-igc-core-2_2.42.6%2B23070_amd64.deb /tmp/intel-runtime/intel-igc-core-2_2.42.6+23070_amd64.deb
ADD https://github.com/intel/intel-graphics-compiler/releases/download/v2.42.6/intel-igc-opencl-2_2.42.6%2B23070_amd64.deb /tmp/intel-runtime/intel-igc-opencl-2_2.42.6+23070_amd64.deb

RUN echo "b2868d6d7958cc5ca28fa9b48a7d39961760ee39ef07673d5b58894dd53d6b4e  /tmp/intel-runtime/intel-ocloc_26.40.40196.8-0_amd64.deb" | sha256sum -c - \
 && echo "96046069e94680b761df6e83ddc64244721178b826e3e1f8af3ca104cfd0c7c2  /tmp/intel-runtime/intel-opencl-icd_26.40.40196.8-0_amd64.deb" | sha256sum -c - \
 && echo "053fceadb84a1d2f9a30a0d1bc9b16424b3f1a90cc49900fcd80aadcf25c8a4d  /tmp/intel-runtime/libigdgmm12_22.10.2_amd64.deb" | sha256sum -c - \
 && echo "d9c86b2bb195baea6936c2708846cd1fceeadc39bc5e167f728858a5ca6fb86e  /tmp/intel-runtime/libze-intel-gpu1_26.40.40196.8-0_amd64.deb" | sha256sum -c - \
 && echo "e20539632f2d99875d8f8b225d5da58b31f98baa54bb47c6328aa98ec473ab9d  /tmp/intel-runtime/intel-igc-core-2_2.42.6+23070_amd64.deb" | sha256sum -c - \
 && echo "7f5593cf828503ffb8450967768bde9209e22753bc576296e846b5a751011fa3  /tmp/intel-runtime/intel-igc-opencl-2_2.42.6+23070_amd64.deb" | sha256sum -c - \
 && dpkg -i /tmp/intel-runtime/*.deb \
 && apt-get check && rm -rf /tmp/intel-runtime && ldconfig

FROM runtime-base AS native-build
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg patch \
 && curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
RUN echo 'deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main' > /etc/apt/sources.list.d/oneapi.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends intel-oneapi-compiler-dpcpp-cpp-2026.2=2026.2.0-863 \
 && rm -rf /var/lib/apt/lists/*
COPY sources/ /sources/
COPY matrix-source/ /matrix-source/
COPY m04-source/ /m04-source/
COPY direct/build-native.sh direct/write-m04-receipt.py /build-tools/
RUN bash /build-tools/build-native.sh
COPY direct/pti/ /build-tools/pti/
RUN bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && python /build-tools/pti/build.py'

FROM runtime-base AS candidate
WORKDIR /opt/exl3xpu
COPY --from=native-build /opt/b70/ /opt/b70/
COPY --from=native-build /native/ /native/
COPY --from=native-build /opt/b70-pti/lib/libpti_view.so.1.2.1 /opt/venv/lib/libpti_view.so.1.2.1
COPY --from=native-build /opt/b70-pti/lib/pti/libpti.so /opt/venv/lib/pti/libpti.so
COPY --from=native-build /sources/onednn/LICENSE /opt/b70-direct/licenses/oneDNN-LICENSE
COPY --from=native-build /sources/xpu-kernels/LICENSE /opt/b70-direct/licenses/vllm-xpu-kernels-LICENSE
COPY --from=native-build /sources/sycl-tla/LICENSE.txt /opt/b70-direct/licenses/SYCL-TLA-LICENSE
COPY --from=native-build /sources/pti/LICENSE /opt/b70-direct/licenses/PTI-LICENSE
COPY direct/runtime/ /opt/exl3xpu/
COPY direct/vllm/ /opt/b70-direct/vllm/
COPY direct/pti/ /opt/b70-direct/pti/
COPY matrix-source/ /opt/exl3xpu/native-matrix-source/
COPY m04-source/ /opt/exl3xpu/m04/
COPY runtime/ /opt/b70-runtime/
COPY direct/build-lock.json direct/source-manifest.json direct/install.py direct/check-image.py /opt/b70-direct/
COPY cow_copy_fix.py /opt/b70-direct/b70_cow_copy_fix.py
COPY b70_mtp_policy.py /opt/b70-direct/b70_mtp_policy.py
ENV LD_LIBRARY_PATH=/opt/b70:/opt/b70/lib:/opt/ucx/lib:/opt/venv/lib:/usr/local/lib
RUN pip install --no-cache-dir --no-deps pyzes==0.1.3 \
 && python /opt/b70-direct/install.py \
 && pip install --no-deps --no-build-isolation -e /opt/exl3xpu \
 && python /opt/b70-direct/check-image.py
ENV MODELS_DIR=/models EXL3_TARGET_RUNTIME=vllm033 EXL3_FIX_SPARSE_GDN_RETIREMENT=0 \
 EXL3_SDPA_CACHE_CAPACITY=64 B70_GPTQ_W4A8_PREFILL=0 B70_ONEDNN_PREFILL=0 \
 B70_ONEDNN_MIXED_ROUTE=0
LABEL org.local.b70.release-status="CANDIDATE_UNQUALIFIED" org.local.b70.experiment="direct-xpu-033"
ENTRYPOINT ["/opt/venv/bin/vllm"]
CMD ["--help"]
