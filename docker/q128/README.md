# Q128 prefill extension

This directory contains the exact Q128/KV32 prefill extension used for the
published single-B70 180 W profile. It replaces only eligible causal prefill
dispatches for the pinned Qwen3.8-27B shape. Decode and every unsupported shape
fall back to the native `vllm-xpu-kernels` path.

`tiles.so` is committed because the public runtime image does not contain the
oneAPI compiler and source trees needed to rebuild it. Its source is
`tiles.cpp` (SHA-256
`c2731c6bfe63bd900d69fdc300ab9c23383ee5b37c8eedd141d9c5b3cc156b3d`);
the adapter `b70_attention.py` has SHA-256
`ae6b78dd316484a16b68aa4d09f674f9d5ca327ebf6b53a08724a8a1e94b9141`,
and the binary SHA-256 is
`f38f23c4535407b6c3f083c5e81c89c7f704bae374571cf2e32f9da13abe0873`.
It was built against the libraries in the pinned vLLM 0.29 XPU base image,
`vllm-xpu-kernels` 0.1.14.1, and the matching SYCL-TLA headers. Do not reuse
the binary with another base image or ABI.

The Python adapter validates the complete tensor shape, dtype, strides and
attention options before dispatch. `variant=1` selects the Q128/subgroup-16
policy. The optional `/evidence/validate` switch compares the first unseen
real activation per query/KV shape against the native kernel.
