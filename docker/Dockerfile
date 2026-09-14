ARG BASE_IMAGE=vllm/vllm-openai-xpu@sha256:96db42e248d48760a4937eb3d04c4878b39d13a9814efea95d510393e097a901
FROM ${BASE_IMAGE}

USER root

# Measured Battlemage userspace. Every package is verified before
# installation; the base image itself is pinned by digest above.
ADD https://github.com/intel/intel-graphics-compiler/releases/download/v2.40.13/intel-igc-core-2_2.40.13%2B22418_amd64.deb /tmp/intel-runtime/intel-igc-core.deb
ADD https://github.com/intel/intel-graphics-compiler/releases/download/v2.40.13/intel-igc-opencl-2_2.40.13%2B22418_amd64.deb /tmp/intel-runtime/intel-igc-opencl.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/intel-ocloc_26.31.39395.13-0_amd64.deb /tmp/intel-runtime/intel-ocloc.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/intel-opencl-icd_26.31.39395.13-0_amd64.deb /tmp/intel-runtime/intel-opencl-icd.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/libigdgmm12_22.10.0_amd64.deb /tmp/intel-runtime/libigdgmm12.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/libze-intel-gpu1_26.31.39395.13-0_amd64.deb /tmp/intel-runtime/libze-intel-gpu1.deb

RUN echo "ebd795e9fddf303a9b24b7f04545d8ddd9ad1f85b3d0cb1166476fab24da6d44  /tmp/intel-runtime/intel-igc-core.deb" | sha256sum -c - \
 && echo "4f990874efc11c3f6091a663b08aef576c4af592dcd8f12e116f8c2fc92d34d9  /tmp/intel-runtime/intel-igc-opencl.deb" | sha256sum -c - \
 && echo "12c5e61ed1dca5cbf38494e280abf88100a451580d57c44f601a17d9727e465e  /tmp/intel-runtime/intel-ocloc.deb" | sha256sum -c - \
 && echo "5a9c9e8fdca8a2f9e22754b1a4618c7babf21d7c3ab3503c680005007c7a8c44  /tmp/intel-runtime/intel-opencl-icd.deb" | sha256sum -c - \
 && echo "6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54  /tmp/intel-runtime/libigdgmm12.deb" | sha256sum -c - \
 && echo "1722943f81b576b9bb8d61016464208f48ce533dc3bf24ad39605293115cc289  /tmp/intel-runtime/libze-intel-gpu1.deb" | sha256sum -c - \
 && dpkg -i /tmp/intel-runtime/*.deb \
 && rm -rf /tmp/intel-runtime

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
COPY q128/b70_attention.py /opt/venv/lib/python3.12/site-packages/b70_attention.py
COPY q128/patch_attention_import.py /tmp/patch_attention_import.py
RUN echo "f38f23c4535407b6c3f083c5e81c89c7f704bae374571cf2e32f9da13abe0873  /opt/b70/tiles.so" | sha256sum -c - \
 && python /tmp/patch_attention_import.py \
 && rm /tmp/patch_attention_import.py

ENTRYPOINT ["vllm", "serve"]
