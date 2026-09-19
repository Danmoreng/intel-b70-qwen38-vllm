# Third-party software and model notice

This repository does not redistribute model weights, Intel driver packages, or
the vLLM container. The build downloads pinned upstream artifacts and verifies
their SHA-256 digests before executing or installing them.

- vLLM is licensed under Apache-2.0.
- `mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` declares Apache-2.0
  and is a quantized derivative of `Qwen/Qwen3.8-27B`.
- The five MTP and prefix-cache patches fetched from
  `SergiioB/intel-arc-pro-b70-inference-cookbook` are covered by that
  project's MIT license. The build pins commit
  `966c593a8b375c4df5173d8d07b6be4db7835fdb`.
- `scripts/context-benchmark.py` and `scripts/generate-exact-prompts.py` are
  adapted from that same MIT-licensed cookbook revision.
- `docker/q128/tiles.cpp` instantiates the public Q128/subgroup-16 policy from
  the pinned `vllm-xpu-kernels`/SYCL-TLA interfaces. The matching prebuilt
  extension is redistributed only for the exact pinned vLLM XPU ABI; source
  and SHA-256 are included in `docker/q128/README.md`.
- `docker/m04/m04.so` is the deployed shared-KV verification extension;
  its local sources and upstream build pins are documented in
  `docker/m04/README.md`. It uses the pinned vLLM-XPU-kernels and SYCL-TLA
  interfaces and is restricted to the same runtime ABI.
- Intel Compute Runtime and Intel Graphics Compiler packages retain their own
  upstream licenses.

Review all upstream licenses before redistribution. This project is an
independent community recipe and is not affiliated with Intel, Qwen, vLLM,
Hugging Face, or the upstream cookbook authors.
