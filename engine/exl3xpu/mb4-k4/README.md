# MB4-DPAS / exact K4 matrix patch

Measured MTP parent: `f1822d3b…`; guarded matrix runtime: `dfd1514f…`. The production manifest pins a child with the same inference payload, adding compiled-source attestation and release identity. No model weights or native binaries are committed. Source inherits the [EXL3 license](../LICENSE).

`source/` contains exactly the six inputs recorded in `native-build-manifest.json`. `mb4-k4-window.patch` is the complete delta from the bootstrap in `../source.tar.gz`. Enable only `EXL3_FLAGS=-DEXL3_DPAS_MB4`. The four-row specialization already existed: the shared selector preserves tuned 24/40/48 blocks. K4 removes only previous-word contributions that vanish after the16-bit mask, preserving the32-lane VNNI layout. No new approximation or quantization.

Rebuild in the pinned native builder `sha256:a78ff8ea07752bb3bc156cd47043386777031f54531075060889f5b2b55b0505` (Torch 2.14 / oneAPI 2026.1.1), with exact oneDNN 3.13.4 include/lib at `/opt/b70`. Mount `source/` read-only at `/source`, a fresh writable directory at `/output`, set workdir `/source`, and execute:

```sh
EXL3_OUT=/output/_C.so EXL3_BUILD_LOG=/output/compiler.log \
EXL3_DNNL_ROOT=/opt/b70 EXL3_FLAGS=-DEXL3_DPAS_MB4 bash scripts/build_ext.sh
```

The build validates ABI, registration and source hashes. A rebuild may have a new binary hash and needs its own gates. `matrix_panel.py` and `check_matrix_xpu.py` provide actual target matrices and the independent bitstream/FP32 oracle with graphs/padding. `prepare_matrix_image.py --build BUILD --numerics NUMERICS_JSON --panel PANEL_JSON --out NEW_CONTEXT` requires same-binary gate receipts and adds guarded library/plugin on the pinned MTP parent. Its output is an experiment, not approval. The generator retains an unused historical `/opt/b70-dspine/` receipt path; no DSpine drafter is loaded.

`release.Dockerfile` adds the exact compiled inputs at `/opt/exl3xpu/native-matrix-source/`. The verifier checks the native library against that actual source tree. Parent runtime/layout guards keep their original source paths unchanged. The manifest pins both guard families, native binary, verifier and compiled inputs. This corrects source attestation without changing model execution.

See [measurements and open follow-ups](../../../benchmarks/results/mtp-matrix-20261110/README.md).
