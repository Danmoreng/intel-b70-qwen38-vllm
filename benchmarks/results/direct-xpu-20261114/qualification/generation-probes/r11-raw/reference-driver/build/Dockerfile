FROM sha256:b37eb3926bade76db626a261b98de6570036fd59d81a3a06e6ab780a87f0918d
ADD https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5/intel-igc-core-2_2.41.5%2B22716_amd64.deb /tmp/intel-runtime/intel-igc-core.deb
ADD https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5/intel-igc-opencl-2_2.41.5%2B22716_amd64.deb /tmp/intel-runtime/intel-igc-opencl.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/intel-ocloc_26.35.39758.10-0_amd64.deb /tmp/intel-runtime/intel-ocloc.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/intel-opencl-icd_26.35.39758.10-0_amd64.deb /tmp/intel-runtime/intel-opencl-icd.deb
ADD https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/libze-intel-gpu1_26.35.39758.10-0_amd64.deb /tmp/intel-runtime/libze-intel-gpu1.deb
RUN echo "0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628  /tmp/intel-runtime/intel-igc-core.deb" | sha256sum -c - \
 && echo "779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058  /tmp/intel-runtime/intel-igc-opencl.deb" | sha256sum -c - \
 && echo "c64bff586edf2bd9b49f3e4c9c2be25e05c43466e4dafa2a2c3948d498c736b7  /tmp/intel-runtime/intel-ocloc.deb" | sha256sum -c - \
 && echo "61712caaddeba3d38e4f79e2a0fb23fea25596ca2d72c3144c6eea2331ec4301  /tmp/intel-runtime/intel-opencl-icd.deb" | sha256sum -c - \
 && echo "c19a641b953d55aebbf1d51bec364a84bf629f985e02fbbe6dc70224c0e88470  /tmp/intel-runtime/libze-intel-gpu1.deb" | sha256sum -c - \
 && dpkg -i /tmp/intel-runtime/*.deb \
 && apt-get check && rm -rf /tmp/intel-runtime && ldconfig
LABEL org.local.b70.release-status="DIAGNOSTIC_ONLY_UNQUALIFIED"
