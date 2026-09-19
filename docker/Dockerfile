ARG BASE_IMAGE=vllm/vllm-openai-xpu@sha256:96db42e248d48760a4937eb3d04c4878b39d13a9814efea95d510393e097a901
FROM ${BASE_IMAGE}

USER root

# Measured Battlemage userspace. Every package is verified before
# installation; the base image itself is pinned by digest above.
ADD https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5/intel-igc-core-2_2.41.5%2B22716_amd64.deb /tmp/intel-runtime/intel-igc-core.deb
ADD https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5/intel-igc-opencl-2_2.41.5%2B22716_amd64.deb /tmp/intel-runtime/intel-igc-opencl.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/intel-ocloc_26.35.39758.10-0_amd64.deb /tmp/intel-runtime/intel-ocloc.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/intel-opencl-icd_26.35.39758.10-0_amd64.deb /tmp/intel-runtime/intel-opencl-icd.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/libigdgmm12_22.10.0_amd64.deb /tmp/intel-runtime/libigdgmm12.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/libze-intel-gpu1_26.35.39758.10-0_amd64.deb /tmp/intel-runtime/libze-intel-gpu1.deb

RUN echo "0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628  /tmp/intel-runtime/intel-igc-core.deb" | sha256sum -c - \
 && echo "779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058  /tmp/intel-runtime/intel-igc-opencl.deb" | sha256sum -c - \
 && echo "c64bff586edf2bd9b49f3e4c9c2be25e05c43466e4dafa2a2c3948d498c736b7  /tmp/intel-runtime/intel-ocloc.deb" | sha256sum -c - \
 && echo "61712caaddeba3d38e4f79e2a0fb23fea25596ca2d72c3144c6eea2331ec4301  /tmp/intel-runtime/intel-opencl-icd.deb" | sha256sum -c - \
 && echo "6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54  /tmp/intel-runtime/libigdgmm12.deb" | sha256sum -c - \
 && echo "c19a641b953d55aebbf1d51bec364a84bf629f985e02fbbe6dc70224c0e88470  /tmp/intel-runtime/libze-intel-gpu1.deb" | sha256sum -c - \
 && dpkg -i /tmp/intel-runtime/*.deb \
 && rm -rf /tmp/intel-runtime \
 && ldconfig

# Correctness and MTP support patches from the pinned public B70 cookbook.
ARG COOKBOOK_COMMIT=966c593a8b375c4df5173d8d07b6be4db7835fdb
ADD https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/${COOKBOOK_COMMIT}/patches/patch_mtp_nightly.py /opt/b70-upstream/patch_mtp_nightly.py
ADD https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/${COOKBOOK_COMMIT}/patches/patch_mtp_boundary.py /opt/b70-upstream/patch_mtp_boundary.py
ADD https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/${COOKBOOK_COMMIT}/patches/patch_fix_accepted_sync.py /opt/b70-upstream/patch_fix_accepted_sync.py
ADD https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/${COOKBOOK_COMMIT}/patches/patch_fix_backward_copy.py /opt/b70-upstream/patch_fix_backward_copy.py
ADD https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/${COOKBOOK_COMMIT}/patches/patch_fix_eagle_drop.py /opt/b70-upstream/patch_fix_eagle_drop.py

RUN echo "4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14  /opt/b70-upstream/patch_mtp_nightly.py" | sha256sum -c - \
 && echo "41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50  /opt/b70-upstream/patch_mtp_boundary.py" | sha256sum -c - \
 && echo "56ff7bdfd093a770db6ae3fa3b0af146fbe8f51022819595d660859251136f07  /opt/b70-upstream/patch_fix_accepted_sync.py" | sha256sum -c - \
 && echo "86fccfdb2d34c1546c3d3972827efd49ee73cdb49bb155d6ae1f4feaa56b64f2  /opt/b70-upstream/patch_fix_backward_copy.py" | sha256sum -c - \
 && echo "fba4708fc6e5d7feaee93bd243129d110fd2d2cde428832a36c44e258f485570  /opt/b70-upstream/patch_fix_eagle_drop.py" | sha256sum -c - \
 && python /opt/b70-upstream/patch_mtp_nightly.py \
 && python /opt/b70-upstream/patch_mtp_boundary.py \
 && python /opt/b70-upstream/patch_fix_backward_copy.py \
 && python /opt/b70-upstream/patch_fix_eagle_drop.py \
 && python /opt/b70-upstream/patch_fix_accepted_sync.py \
 && rm -rf /opt/b70-upstream

# Local performance patches. They only alter the speculative draft; target
# GPTQ weights and target verification stay unchanged. Supported for TP=1.
COPY patches/ /opt/b70-local-patches/
RUN python /opt/b70-local-patches/patch_draft_lmhead_int4.py \
 && python /opt/b70-local-patches/patch_draft_mtp_int4.py \
 && rm -rf /opt/b70-local-patches

# Exact Q128/KV32 prefill extension qualified with this pinned vLLM/XPU ABI.
# The adapter fails closed to the native attention path for every unsupported
# tensor shape, dtype, stride or option. Variant 1 is Q128/subgroup-16.
COPY q128/tiles.so /opt/b70/tiles.so
# Combined production adapter: Q128 prefill and M04 shared-KV verification.
COPY m04/m04.so /opt/b70/m04.so
COPY m04/b70_attention.py /opt/venv/lib/python3.12/site-packages/b70_attention.py
COPY q128/patch_attention_import.py /tmp/patch_attention_import.py
RUN echo "f38f23c4535407b6c3f083c5e81c89c7f704bae374571cf2e32f9da13abe0873  /opt/b70/tiles.so" | sha256sum -c - \
 && echo "784916abd42b614794becac634a6a154f854c73d6dfcb1a50cd939b563020c6a  /opt/b70/m04.so" | sha256sum -c - \
 && echo "d288bb54f52f5be166e851b2d7862da81c041bd7eff6b90e6eaca68c92333b64  /opt/venv/lib/python3.12/site-packages/b70_attention.py" | sha256sum -c - \
 && python /tmp/patch_attention_import.py \
 && rm /tmp/patch_attention_import.py

ENTRYPOINT ["vllm", "serve"]
